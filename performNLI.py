import unidecode
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import gc

from tqdm import tqdm

# try eval source vs reference, to see how good the multilingual NLI is?

def runNLIStage(inputList):
  print("RUNNING NLI:")
  
  model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" # has best correlation on example
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

  errors = 0
  outputList = []
  for pair in tqdm(inputList):
    try:
      ref = unidecode.unidecode(pair['ref']).strip().split('\n')
      pred = unidecode.unidecode(pair['output']).strip().split('\n')

      scores = []
      outScore = 0

      if len(ref) != len(pred):
        raise Exception(f"ref and pred have differing line counts! [ref: {len(ref)} | pred: {len(pred)}]")
      
      for x in range(len(ref)):
        input = tokenizer(ref[x], pred[x], truncation=True, return_tensors="pt")

        output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        prediction = {name: float(pred) for pred, name in zip(prediction, label_names)}
        # scores_out.append(prediction['entailment'])
        score = 0
        if 'entailment' in prediction:
          score = prediction['entailment']
        # if 'neutral' in prediction: # as bad as a contradiction? as in, it has a different emaning, so count against equally?
        #   score -= prediction['neutral']
        if 'contradiction' in prediction:
          score -= prediction['contradiction']
        scores.append(score)
        outScore += score
      
      outScore = outScore/len(ref)

      outputList.append({'id': pair['id'], 'ref': pair['ref'], 'output': pair['output'], 'score': outScore, 'scores': scores})

    except:
      errors += 1
    
  del model
  del tokenizer

  gc.collect()
  torch.cuda.empty_cache()

  return outputList, errors

  

#coref needs to make smarter matches, will help NLI model. Or find a better NLI model...   
# try NLI with no coref, but a few reference lines as context.      

#match back up to doc?.. maybe take off scores for missed clusters too?
# ref_disambig = getDisambig(named_ref, ref_tokenized)

#when doing eval, make sure translations have identical line counts as source and reference
#also when doing eval, replace all the funky characterquotes, periods, etc with the proper ascii versions

# print(ref_disambig)