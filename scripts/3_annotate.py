import json

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.spatial import distance
import tqdm


satellites = []

with open('data/spotGEO/train_anno.json') as f:
    for ann in json.load(f):
        for i, coords in enumerate(ann['object_coords']):
            satellites.append({
                'sequence': ann['sequence_id'],
                'frame': ann['frame'],
                'satellite': i + 1,
                'r': int(coords[1] + .5),
                'c': int(coords[0] + .5),
            })

satellites = pd.DataFrame(satellites)
satellites = satellites.set_index(['sequence', 'frame', 'satellite'])
satellites.head()

df = pd.read_pickle('data/interesting.pkl')
labels = []

for (sequence, frame), g in tqdm.tqdm(df.query('part == "train"').groupby(['sequence', 'frame'])):

    try:
        sats = satellites.loc[sequence, frame]
    except KeyError:
        continue

    # Compute the distance between each satellite and each interesting location,
    # thus forming a bipartite graph
    centers = g[['r', 'c']]
    distances = distance.cdist(sats, centers, metric='chebyshev')

    # Guess which locations correspond to which satellites
    row_ind, col_ind = optimize.linear_sum_assignment(distances)

    # Each satellite is assigned, but some of them may be too distant to be likely
    likely = distances[row_ind, col_ind] <= 2

    is_satellite = np.full(len(centers), False, dtype=bool)
    is_satellite[col_ind[likely]] = True
    labels.append(pd.DataFrame({
        'part': 'train',
        'sequence': sequence,
        'frame': frame,
        'r': g['r'],
        'c': g['c'],
        'is_satellite': is_satellite
    }))

labels = pd.concat(labels).set_index(['part', 'sequence', 'frame', 'r', 'c'])
df = df.join(labels, on=labels.index.names)
df['is_satellite'] = df['is_satellite'].fillna(False)
df.loc[df['part'] == 'test', 'is_satellite'] = np.nan
df.to_pickle('data/interesting.pkl')

print(f'Recall is {df.is_satellite.sum() / len(satellites):.2%}')
