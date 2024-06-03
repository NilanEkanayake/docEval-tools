from localModel import generateLocal
from performCoref import runCorefStage
from performNLI import runNLIStage

import argparse
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument("--lora", help="lora path")
parser.add_argument("--dataset", help="dataset path")
args = parser.parse_args()

with jsonlines.open(args.dataset) as reader:
    data = list(reader)

    for x in range(len(data)):
        data[x]['id'] = x

    modelOutputs = generateLocal(data, args.lora)
    modelLosses = len(data) - len(modelOutputs)

    corefOutputs, corefErrors = runCorefStage(modelOutputs)
    corefLosses = len(modelOutputs) - len(corefOutputs)

    NLIOutput, NLIErrors = runNLIStage(corefOutputs)
    NLILosses = len(corefOutputs) - len(NLIOutput)

    totalScore = 0
    minScore = 1000000
    maxScore = -1000000

    for item in NLIOutput:
        totalScore += item['score']
        if item['score'] < minScore:
            minScore = item['score']
        if item['score'] > maxScore:
            maxScore = item['score']
    totalScore = totalScore/len(NLIOutput)

    print('-'*50)
    print(f"MODEL LOSSES: {modelLosses} | COREF LOSSES: {corefLosses} | NLI LOSSES: {NLILosses}")
    print('-'*50)
    print(f"MIN SCORE: {round(minScore*100, 2)}")
    print(f"MAX SCORE: {round(maxScore*100, 2)}")
    print(f"AVG SCORE: {round(totalScore*100, 2)}")
