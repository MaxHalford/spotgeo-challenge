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

Coordinate starts at the top left. x is the column, and y is the row.

## Steps to reproduce

```sh
python setup.py build_ext --inplace
```
