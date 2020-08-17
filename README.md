spotgeo-challenge


https://kelvins.esa.int/spot-the-geo-satellites/data/

## Data

- The coordinates from train_anno.json have as reference the bottom left corner, whereas images start at the top left (?????)
- scikit-image starts at the top left

- 1280 train sequences
- 5120 test sequences

## Strategy

1. For each image, determine which pixels are "interesting".
2. Annotate each interesting region.
3. Calculate features for each interesting region.
4. Train.
5. Postprocess predictions across a sequence.

   precision    recall  f1-score   support

         0.0     0.9993    0.9999    0.9996   3928795
         1.0     0.9316    0.7217    0.8133      9813

    accuracy                         0.9992   3938608
   macro avg     0.9655    0.8608    0.9065   3938608
weighted avg     0.9991    0.9992    0.9991   3938608

precision    recall  f1-score   support

         0.0     0.9994    0.9998    0.9996   3928795
         1.0     0.9003    0.7583    0.8232      9813

    accuracy                         0.9992   3938608
   macro avg     0.9498    0.8790    0.9114   3938608
weighted avg     0.9991    0.9992    0.9992   3938608

**Difficult sequences to spot**

- 102
- 104
- 112
- 126
- 140
- 146!
- 155
- 157
- 165
- 172
- 182
- 184
- 188
- 193
- 197
- 205
- 216
- 221
- 233
- 242
- 245
- 252
- 283
- 286
- 291

## Steps to reproduce

```sh
python setup.py build_ext --inplace
```
