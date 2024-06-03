Tools to perform reference-based document translation evaluation.

To evaluate a sangle translation against a reference:
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
Clone and place in the sample folder as the .py files.

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
