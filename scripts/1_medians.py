import os
import pathlib

import numpy as np
from PIL import Image
from scipy import ndimage
import tqdm


for part in ['train', 'test']:

    for sequence in tqdm.tqdm(list(pathlib.Path(f'data/spotGEO/{part}').glob('*')), position=0):
        os.makedirs(f'data/medians/{part}/{sequence.name}', exist_ok=True)

        for frame in sequence.glob('*.png'):

            img = Image.open(frame)
            pixels = np.asarray(img)
            img.close()
            med = ndimage.median_filter(pixels, size=(16, 12))
            np.save(f'data/medians/{part}/{sequence.name}/{frame.stem}.npy', med)
