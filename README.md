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

## To do

- imblearn instead of weighting instances
- gaussian blur might for finding interesting points
- consider rectangle regions instead of squares
- check if playing with the classification threshold influences the F1 score?

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
