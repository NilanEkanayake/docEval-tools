Tools to perform reference-based document translation evaluation.

To evaluate a single translation against a reference:
```
python3 evaluateSample.py --reference reference.txt --hypothesis hypothesis.txt
```

Required packages:
```
https://github.com/urchade/GLiNER
nltk
tqdm
spacy
unidecode
torch
transformers
```

Also required is https://github.com/shtoshni/fast-coref.
Clone and setup according to the instructions in the repo's readme (make sure to download the models). Place the 'fast_coref' and 'models' folders in the same directory as the python files.

Before use, edit fast_coref/src/inference/model_inference.py:

Change:
```
cur_cluster.append(
    (
        (ment_start, ment_end),
        " ".join(
            orig_tokens[
                subtoken_map[ment_start] : subtoken_map[ment_end] + 1
            ]
        ),
    )
)
```
To:
```
cur_cluster.append(
    (
        (subtoken_map[ment_start], subtoken_map[ment_end]), " ".join(orig_tokens[subtoken_map[ment_start] : subtoken_map[ment_end] + 1]),
    )
)
```
