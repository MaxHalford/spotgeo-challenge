import pathlib

import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure
import tqdm

regions = []

for part in ['train']:

    for sequence_path in tqdm.tqdm(list(pathlib.Path(f'data/spotGEO/{part}').glob('*')), position=0):
        if sequence_path.name == '.DS_Store': continue
        sequence = int(sequence_path.name)

        for frame_path in sequence_path.glob('*.png'):
            frame = int(frame_path.stem)

            # Access the necessary data
            img = Image.open(frame_path)
            pixels = np.asarray(img)
            img.close()
            med = np.load(f'data/medians/{part}/{sequence}/{frame}.npy')

            # Determine which pixels are brighter than their surroundings
            interesting = pixels > med + 6

            # Label each pixel
            labels = measure.label(interesting)

            # Group identically labeled pixels into regions
            rs = measure.regionprops(label_image=labels, intensity_image=pixels)
            centers = np.asarray([
                r.coords[np.take(pixels, np.ravel_multi_index(r.coords.T, pixels.shape)).argmax()]
                for r in rs
            ])

            for region, (r, c) in zip(rs, centers):
                regions.append({
                    'part': part,
                    'sequence': sequence,
                    'frame': frame,
                    'r': r,
                    'c': c,
                    'area': region.area,
                    'extent': region.extent,
                    'perimeter': region.perimeter,
                    'diameter': region.equivalent_diameter,
                    'eccentricity': region.eccentricity,
                    'solidity': region.solidity
                })

df = pd.DataFrame.from_records(regions)
df.to_pickle('data/interesting.pkl')
