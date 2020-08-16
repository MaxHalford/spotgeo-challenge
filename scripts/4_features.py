import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
import tqdm


def region(img: np.ndarray, r: int, c: int, w: int):
    """Returns the square of length width with (x, y) being at the center."""
    return img[
        max(r - w, 0) : min(r + w + 1, img.shape[0]),
        max(c - w, 0) : min(c + w + 1, img.shape[1])
    ]


def extract_features(img, r, c):

    val = img[r, c]
    features = {'pixel_intensity': val}

    for w in [1, 3, 5, 7]:
        pixels = region(img, r, c, w).ravel()
        features.update({
            f'{w}x{w}_mean': val - pixels.mean(),
            f'{w}x{w}_std': pixels.std(),
            f'{w}x{w}_min': val - pixels.min(),
            f'{w}x{w}_max': val - pixels.max(),
            f'{w}x{w}_entropy': stats.entropy(pixels),
            f'{w}x{w}_kurtosis': stats.kurtosis(pixels),
            f'{w}x{w}_skew': stats.skew(pixels),
        })

    return features


df = pd.read_pickle('data/interesting.pkl')

features = []

# There should be 32000 frames (5 * 1280 + 5 * 5120)
for (part, sequence, frame), locations in tqdm.tqdm(df.groupby(['part', 'sequence', 'frame'])):

    img = np.asarray(Image.open(f'data/spotGEO/{part}/{sequence}/{frame}.png')).astype(np.float32)

    for _, location in locations.iterrows():

        r = int(location['r'])
        c = int(location['c'])

        features.append({
            'part': part,
            'sequence': sequence,
            'frame': frame,
            'r': r,
            'c': c,
            **extract_features(img, r=r, c=c)
        })

features = pd.DataFrame.from_records(features)
features = features.set_index(['part', 'sequence', 'frame', 'r', 'c'])
df.join(features, on=features.index.names).to_pickle('interesting.pkl')
