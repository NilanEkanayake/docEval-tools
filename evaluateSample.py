from performCoref import runCorefStage
from performNLI import runNLIStage

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--reference", help="path to reference (txt)")
parser.add_argument("--hypothesis", help="path to hypothesis (txt) - needs same line count as reference text")
args = parser.parse_args()

with open(args.reference) as ref, open(args.hypothesis) as hyp:
    refText = ref.read().strip()
    hypText = hyp.read().strip()

    toCoref = [{'ref': refText, 'output': hypText, 'id': 1}]

    corefOutputs, corefErrors = runCorefStage(toCoref)

    NLIOutput, NLIErrors = runNLIStage(corefOutputs)


    score = NLIOutput[0]['score']

    print('-'*50)
    for entry in zip(refText.split('\n'), hypText.split('\n'), NLIOutput[0]['scores']):
        print(f"REF: {entry[0]} | HYP: {entry[1]} | SCORE: {round(entry[2]*100, 2)}")

    print('-'*50)
    print(f"AVG SCORE: {round(score*100, 2)}")

