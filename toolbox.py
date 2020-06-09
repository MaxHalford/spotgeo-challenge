import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import ImageGrid


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
