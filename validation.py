# Validation and Scoring tool for the spotGEO competition on kelvins.esa.int.

# Imports
import numpy as np
import json
import jsonschema
import sys

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from zipfile import ZipFile

# Constants
min_seq_number = 1
max_seq_number = 5120
frames_per_sequence = 5
img_width  = 639.5
img_height = 479.5
min_size = -0.5
max_number_of_objects = 30


# Submissions have to follow the following schema
schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "sequence_id": { "type" : "integer",
                             "minimum": min_seq_number,
                             "maximum": max_seq_number},
            "frame":       { "type" : "integer",
                             "minimum": 1,
                             "maximum": frames_per_sequence},
            "num_objects": { "type" : "integer",
                             "minimum": 0,
                             "maximum": max_number_of_objects},
            "object_coords": { "type" : "array",
                               "items": {
                                   "type": "array",
                                   "items": [ {"type": "number",
                                               "minimum": min_size,
                                               "maximum": img_width },
                                              {"type": "number",
                                               "minimum": min_size,
                                               "maximum": img_height } ]
                                   }
                             }
        },
        "required": ["sequence_id", "frame", "num_objects", "object_coords"]
    },
}

# Helper functions
def flat_to_hierarchical(labels):
    """ Transforms a flat array of json-objects to a hierarchical python dict, indexed by
        sequence number and frame id. """
    seqs = dict()
    for label in labels:
        seq_id = label['sequence_id']
        frame_id = label['frame']
        coords = label['object_coords']

        if seq_id not in seqs.keys():
            seqs[seq_id] = defaultdict(dict)
        seqs[seq_id][frame_id] = np.array(coords)

    return seqs


def score_frame(X, Y, tau=10, eps=3):
    """ Scoring Prediction X on ground-truth Y by linear assignment. """
    if len(X) == 0 and len(Y) == 0:
        # no objects, no predictions means perfect score
        TP, FN, FP, sse = 0, 0, 0, 0
    elif len(X) == 0 and len(Y) > 0:
        # no predictions but objects means false negatives
        TP, FN, FP, sse = 0, len(Y), 0, len(Y) * tau**2
    elif len(X) > 0 and len(Y) == 0:
        # predictions but no objects means false positives
        TP, FN, FP, sse = 0, 0, len(X), len(X) * tau**2
    else:
        # compute Euclidean distances between prediction and ground truth
        D = cdist(X, Y)

        # truncate distances that violate the threshold
        D[D > tau] = 1000

        # compute matching by solving linear assignment problem
        row_ind, col_ind = linear_sum_assignment(D)
        matching = D[row_ind, col_ind]

        # true positives are matches within the threshold
        TP = sum(matching <= tau)

        # false negatives are missed ground truth points or matchings that violate the threshold
        FN = len(Y) - len(row_ind) + sum(matching > tau)

        # false positives are missing predictions or matchings that violate the threshold
        FP = len(X) - len(row_ind) + sum(matching > tau)

        # compute truncated regression error
        tp_distances = matching[matching < tau]
        # truncation
        tp_distances[tp_distances < eps] = 0
        # squared error with constant punishment for false negatives and true positives
        sse = sum(tp_distances) + (FN + FP) * tau**2

    return TP, FN, FP, sse


def score_sequence(X, Y, tau=10, eps=3):
    # check that X and Y cover all 5 frames
    assert set(X.keys()) == set(Y.keys())

    frame_scores = [score_frame(X[k], Y[k], tau=tau, eps=eps) for k in X.keys()]
    TP = sum([x[0] for x in frame_scores])
    FN = sum([x[1] for x in frame_scores])
    FP = sum([x[2] for x in frame_scores])
    sse = sum([x[3] for x in frame_scores])

    mse = 0 if (TP + FN + FP) == 0 else sse / (TP + FN + FP)
    return TP, FN, FP, mse


def score_sequences(X, Y, tau=10, eps=3, taboolist=[]):
    """ scores a complete submission except sequence_ids that are listed
        in the taboolist. """
    # check that each sequence has been predicted
    #assert set(X.keys()) == set(Y.keys())

    # we filter the identifiers from the taboolist
    identifiers = set(X.keys()) - set(taboolist)

    # compute individual sequence scores
    seq_scores = [score_sequence(X[k], Y[k], tau=tau, eps=eps) for k in identifiers]
    TP = sum([x[0] for x in seq_scores])
    FN = sum([x[1] for x in seq_scores])
    FP = sum([x[2] for x in seq_scores])
    mse = sum([x[3] for x in seq_scores])

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)

    return precision, recall, F1, mse


def compute_score(predictions, labels):
    """ Scores a submission `predictions` against ground-truth `labels`. Does
    not perform any validation and expects `predictions` and `labels` to be
    valid paths to .json-files. """
    with open(predictions, 'rt') as fp:
        predictions_h = flat_to_hierarchical(json.load(fp))

    with open(labels, 'rt') as fp:
        labels_h = flat_to_hierarchical(json.load(fp))

    precision, recall, F1, mse = score_sequences(predictions_h, labels_h)

    return (1 - F1, mse)


def validate_json(labels):
    """ Valides whether `labels` follow the required formats to be accepted
        for computing a score. """
    # 1. Check whether the json follows correct input formats
    jsonschema.validate(labels, schema)

    # 2. jsonschema is not powerful enough to appropriately check for duplicates
    identifiers = [(label['sequence_id'], label['frame']) for label in labels]
    if not len(set(identifiers)) == len(identifiers):
        raise ValueError('Error. You have duplicates in your submission. Make sure each combination of sequence_id and frame is unique.')

    # 3. We need an identifier for each sequence and frame combination
    needed_identifiers = {(i,j) for i in range(min_seq_number, max_seq_number + 1) for j in range(1, frames_per_sequence + 1)}
    missing_identifiers = needed_identifiers - set(identifiers)

    if len(missing_identifiers) > 0:
        raise ValueError(f'Error. Your submission is missing {len(missing_identifiers)} entries')

    # 4. Make sure the number of predicted objects corresponds to the correct array dimensions
    for label in labels:
        if len(label['object_coords']) != label['num_objects']:
            raise ValueError('Error. You indicated to predict {:d} objects, but give coordinates for {:d} in sequence {:d}, frame {:d}.'.format(label['num_objects'], len(label['object_coords']), label['sequence_id'], label['frame']))

    # The file validates successfully
    return True


# This script can run from the command line: python my_anno.json true_labels.json
if __name__ == '__main__':
    if len(sys.argv)  not in [2, 3]:
        print('Usage: \n\tValidation: python my_anno.json\n\tscoring: python my_anno.json true_labels.json')
    else:

        if len(sys.argv) == 3:
            score, mse = compute_score(sys.argv[1], sys.argv[2])
            print('Score: {:0.6f}, (MSE: {:0.6f})'.format(score, mse))
