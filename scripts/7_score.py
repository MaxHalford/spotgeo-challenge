import itertools
import subprocess
import zipfile

import pandas as pd
from scipy import stats
import tqdm

import toolbox

THRESHOLD = .29  # .32 0.138132, (MSE: 20551.285897) (last commit is .42)

oof = pd.read_pickle('data/oof.pkl')
oof = oof[oof > THRESHOLD].sort_index()
toolbox.save_predictions(oof, path='oof.json', n_sequences=1280)
subprocess.run('python validation.py oof.json data/spotGEO/train_anno.json', shell=True)

y_test = pd.read_pickle('data/y_test.pkl')
y_test = y_test.mean(axis='columns')
y_test = y_test[y_test > THRESHOLD].sort_index()
toolbox.save_predictions(oof, path='sub.json', n_sequences=5120)
with zipfile.ZipFile('sub.zip', mode='w') as f:
    f.write('sub.json')


class Trajectory:

    def __init__(self, frames, r, c):
        self.r_linreg = stats.linregress(x=frames, y=r)
        self.c_linreg = stats.linregress(x=frames, y=c)

    @property
    def good_fit(self):
        return abs(self.r_linreg.rvalue) > .999 and abs(self.c_linreg.rvalue) > .999

    def get_path(self):
        return [
            [
                self.r_linreg.slope * frame + self.r_linreg.intercept,
                self.c_linreg.slope * frame + self.c_linreg.intercept,
            ]
            for frame in range(1, 6)
        ]

def find_trajectory(frames):

    # We first consider all the trajectories that are formed with 3, 4, or 5 points from different
    # frame. We fit a line to every possible point combination. We stop if we find a "good" line.
    # We can then remove the involved points and call this function once again.
    for k in range(5, 2, -1):
        for frame_combo in itertools.combinations(range(5), k):
            for coord_combo in itertools.product(*[range(len(frames[f])) for f in frame_combo]):
                r = [frames[i][j][0] for i, j in zip(frame_combo, coord_combo)]
                c = [frames[i][j][1] for i, j in zip(frame_combo, coord_combo)]
                traj = Trajectory(frames=[f + 1 for f in frame_combo], r=r, c=c)
                if traj.good_fit:
                    return frame_combo, coord_combo, traj

    # Now we look at pairs of points in adjacent frames. We assume that two point from different
    # frames form a trajectory if their distance looks likely.
    for i1 in range(4):
        for i2 in range(i1 + 1, min(i1 + 3, 5)):
            for j1, j2 in itertools.product(range(len(frames[i1])), range(len(frames[i2]))):
                r1, c1 = frames[i1][j1]
                r2, c2 = frames[i2][j2]
                dist = abs(r1 - r2) + abs(c1 - c2)
                if 18 * (i2 - i1) <= dist <= 70 * (i2 - i1):
                    traj = Trajectory(frames=[i1 + 1, i2 + 1], r=[r1, r2], c=[c1, c2])
                    return (i1, i2), (j1, j2), traj

    return None

def expand_predictions(predictions):

    trajectories = []

    frames = [[] for _ in range(5)]
    for frame, r, c in predictions.index:
        frames[frame - 1].append([r, c])

    while frames:

        result = find_trajectory(frames)
        if result is None:
            break

        frame_combo, coord_combo, traj = result
        trajectories.append(traj)
        for i, j in zip(frame_combo, coord_combo):
            del frames[i][j]

    if not trajectories:
        return pd.DataFrame()

    expansion = pd.concat([
        pd.DataFrame(traj.get_path(), columns=['r', 'c'])
        for traj in trajectories
    ])
    expansion.index.name = 'frame'
    expansion.index += 1

    return expansion

oof_pp = pd.concat([
    expand_predictions(g.droplevel('sequence')).assign(sequence=sequence)
    for sequence, g in tqdm.tqdm(oof.groupby('sequence'), position=0)
])
oof_pp.index.name = 'frame'
oof_pp = oof_pp.reset_index().set_index(['sequence', 'frame'])
oof_pp = oof_pp.assign(is_satellite=True).set_index(['r', 'c'], append=True)
toolbox.save_predictions(
    predictions=oof_pp['is_satellite'].sort_index(),
    path='oof_pp.json',
    n_sequences=1280
)
subprocess.run('python validation.py oof_pp.json data/spotGEO/train_anno.json', shell=True)


y_test_pp = pd.concat([
    expand_predictions(g.droplevel('sequence')).assign(sequence=sequence)
    for sequence, g in tqdm.tqdm(y_test.groupby('sequence'), position=0)
])
y_test_pp.index.name = 'frame'
y_test_pp = y_test_pp.reset_index().set_index(['sequence', 'frame'])
y_test_pp = y_test_pp.assign(is_satellite=True).set_index(['r', 'c'], append=True)
toolbox.save_predictions(
    predictions=y_test_pp['is_satellite'].sort_index(),
    path='sub_pp.json',
    n_sequences=5120
)
with zipfile.ZipFile('sub_pp.zip', mode='w') as f:
    f.write('sub_pp.json')
