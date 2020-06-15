import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import ImageGrid
import tqdm


def save_predictions(predictions, n_sequences, path):
    """

    predictions should be a dataframe with the following structure:

    sequence  frame  r   c
    1         1      7   339    False
                     10  264     True
                     18  40      True
                     20  462    False
                     26  65      True

    """

    sub = []

    for sequence in tqdm.tqdm(range(1, n_sequences + 1), position=0):

        for frame in range(1, 6):


            try:
                g = predictions.loc[sequence, frame]
            except KeyError:
                sub.append({
                    'sequence_id': int(sequence),
                    'frame': int(frame),
                    'num_objects': 0,
                    'object_coords': []
                })
                continue

            coords = []

            for (*_, r, c), is_sat in g.iteritems():
                if is_sat and 0 <= r <= 480 and 0 <= c <= 640:
                    coords.append([c - .5, r - .5])

            sub.append({
                'sequence_id': int(sequence),
                'frame': int(frame),
                'num_objects': len(coords),
                'object_coords': coords
            })

    with open(path, 'w') as f:
        json.dump(sub, f)


def viz_sequence(seq, df):
    """Displays a sequence."""

    # Make one image for each frame
    fig = plt.figure(figsize=(15, 10 * 5)) # 5 frames in each sequence
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(5, 1),
        axes_pad=.4,
    )

    frames = df.loc[seq]
    trajectories = []

    for frame, sats in frames.groupby('frame'):

        # Sky
        ax = grid[frame - 1]
        ax.imshow(train[seq, frame].T, origin='lower')
        ax.set_title(f'Sequence #{seq}, frame #{frame}')

        # Current positions
        for _, sat in sats.iterrows():
            ax.scatter(sat['x'].astype(int), sat['y'].astype(int), s=100, facecolors='none', edgecolors='red', linewidths=2)

        # Old positions
        for old_sats in trajectories:
            for _, sat in old_sats.iterrows():
                ax.scatter(sat['x'].astype(int), sat['y'].astype(int), s=100, facecolors='none', edgecolors='yellow', linewidths=2)
        trajectories.append(sats)

        ax.set_xlim(-.5, 639.5)
        ax.set_ylim(-.5, 479.5)

    return fig
