import sys
sys.path.append('fast_coref/src')

import unidecode

from nltk.translate.chrf_score import sentence_chrf

from inference.model_inference import Inference
from gliner import GLiNER

import torch
import gc

from tqdm import tqdm

import spacy
nlp = spacy.load('en_core_web_sm')

# split NER input by line


def getClusters(input, min_cluster_len, inference_model):
  output = inference_model.perform_coreference(input)
  result = []
  newlines = []
  for x in range(len(output['tokenized_doc']['orig_tokens'])):
    if output['tokenized_doc']['orig_tokens'][x] == '\n':
        newlines.append(x)
  newlines.append(len(output['tokenized_doc']['orig_tokens'])) # account for last line

  for cluster in output["clusters"]:
    if len(cluster) >= min_cluster_len:
      temp = []
      # shuffle cluster from long match to short to enable better nested matching
      cluster = sorted(cluster, key = lambda x: int(x[0][1]-x[0][0]), reverse=True)
      for item in cluster:
        original = True
        for previous in temp:
          if item[0][0] >= previous[0][0] and item[0][1] <= previous[0][1]:
              original = False
              break
        if original:  
          matchInLine = 0
          for x in range(len(newlines)):
              if item[0][1] < newlines[x]:
                matchInLine = x
                break
          temp.append([[item[0][0], item[0][1]], item[1], matchInLine])
      result.append(temp)
  return result, output['tokenized_doc']['orig_tokens']


def nameClusters(clusters, ner_model, ner_labels):
  out = []
  missCount = 0 # not needed, cause just may be a lot of ltitle misses? Maybe count how long each missed cluster is too.
  for x in range(len(clusters)):
    # seen = ["he", "him", "his" "she", "her", "hers", "they", "them", "theirs", "we", "our", "us", "you", "yours" "it"] # don't NER pronouns
    seen = ["he", "him", "his" "she", "her", "hers", "you", "yours", "anything", "everything", "something", "nothing"]
    # seen = []
    names = []
    for item in clusters[x]:
        if item[1].lower() not in seen:
          seen.append(item[1].lower())
          names.append(item[1].lower())

    if names:
      entities = ner_model.predict_entities(", ".join(names), ner_labels, threshold=0.5)
      if len(entities) == 1:
        out.append({'ner': entities[0], 'clusters': clusters[x]})
      elif len(entities) > 1:
        entities = sorted(entities, key = lambda x: int(x['score']*100), reverse=True)[0] # get longest, and thus most original?
        out.append({'ner': entities, 'clusters': clusters[x]})
      else:
        temp = []
        for name in names:
          # also gets words that end with below matches, so need smarter filter that checks for space or start of string.
          if name.startswith(('a ', 'the ', 'my ', 'an ', 'his ', 'her ', 'their ', 'its ', 'your ', 'our ')): # get entities 
            temp.append(name)
        if temp:
          match = sorted(temp, key = lambda x: len(x), reverse=False)[0] # get shortest entity name
          out.append({'ner': {'text': match}, 'clusters': clusters[x]})
        else:
          # print("MISS")
          # print(clusters[x])
          missCount += 1
    else:
      # print("MISS")
      # print(clusters[x])
      missCount +=1
  return out, missCount


def getDisambig(named, tokenized):
  for item in named:
    name = item['ner']['text'].lower()
    indexes = []
    for cluster in item['clusters']:
      indexes.append(cluster[0])

    for index in indexes:
      if index[1]-index[0] == 0:
        tokenized[index[0]] = name
      else:
        tokenized[index[0]] = name
        for x in range(1, index[1]-index[0]+1):
          tokenized[index[0]+x] = ''

  out_string = ""
  prev_token = ''
  for out_token in tokenized:
    if out_token != '':
      out_token = out_token.replace("’", "'")
      out_token = out_token.replace('“', '"')
      out_token = out_token.replace('”', '"')
      if (len(out_token) == 1 and not out_token.isalpha()) or "'" in out_token or prev_token == '"' or '…' in prev_token or '...' in prev_token or prev_token == '\n':
        out_string += out_token
      else:
        out_string += ' ' + out_token
      prev_token = out_token
  return out_string.strip()

   
def uniteNames(refString, predString, ner_model, ner_labels):
  refList = refString.strip().split('\n')
  refEntities = []
  for line in refList:
    doc = nlp(line)
    for sent in doc.sents:
      result = ner_model.predict_entities(sent.text, ner_labels, threshold=0.6)
      if result:
        refEntities += result
  refNames = []
  outString = predString
  for item in refEntities:
    if item['text'] not in refNames:
      refNames.append(item['text'])

  names = []
  predList = predString.strip().split('\n')
  predEntities = []
  for line in predList:
    doc = nlp(line)
    for sent in doc.sents:
      result = ner_model.predict_entities(sent.text, ner_labels, threshold=0.6)
      if result:
        predEntities += result

  for item in predEntities:
    if item['text'] not in names:
      names.append(item['text'])

  for name in names:
    topScore = 0
    topScoreText = ""
    for refName in refNames:
      score = sentence_chrf(refName.lower(), name.lower())*100
      if score > topScore:
        topScore = score
        topScoreText = refName
    if topScore > 10:
      outString = outString.replace(name, topScoreText)
  
  return refString, outString



def runCorefStage(inputList):
  print("RUNNING COREF:")

  inference_model = Inference("models/joint_best", encoder_name="shtoshni/longformer_coreference_joint")
  ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
  ner_labels = ['character', 'object', 'location', 'organization']

  errors = 0
  outputList = []

  for pair in tqdm(inputList):
    try:
      refString = unidecode.unidecode(pair['ref']).replace('|END|', '').strip()
      predString = unidecode.unidecode(pair['output']).replace('|END|', '').strip()

      refString, predString = uniteNames(refString, predString, ner_model, ner_labels)

      # run batched with all pairs at once?
      ref, ref_tokenized = getClusters(f"{refString}\n|END|\n{predString}\n|END|", 2, inference_model)

      named_ref, missCount = nameClusters(ref, ner_model, ner_labels)

      ref_disambig = getDisambig(named_ref, ref_tokenized).split('\n')

      sampleLineCount = refString.count('\n')

      out_ref = ""
      out_pred = ""

      for x in range(len(ref_disambig)):
        if ref_disambig[x] == '|END|':
          if int(x / sampleLineCount) == 1:
            out_ref = "\n".join(ref_disambig[x-sampleLineCount-1:x])
          else:
            out_pred = "\n".join(ref_disambig[x-sampleLineCount-1:x])
      outputList.append({'id': pair['id'], 'ref': out_ref, 'output': out_pred}) # id should be the position of the item in the inputList, used to track
    except:
      errors += 1

  del inference_model
  del ner_model

  gc.collect()
  torch.cuda.empty_cache()

  return outputList, errors

    

