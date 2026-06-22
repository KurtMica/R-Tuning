import json
import sys
import os
import argparse
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluate import ESTIMATORS


def calculate_scores(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    labels = [s[0] for s in data]
    n_scores = len(data[0]) - 1   # number of score columns (excluding correct)

    scores = {
        "predict_conf": average_precision_score(labels, [s[1] for s in data]),
        "sure_prob":    average_precision_score(labels, [s[2] for s in data]),
        "combined":     average_precision_score(labels, [0.5 * s[1] + 0.5 * s[2] for s in data]),
    }

    if n_scores > 2:
        for i, name in enumerate(ESTIMATORS.keys()):
            scores[name] = average_precision_score(labels, [s[i + 3] for s in data])

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Average Precision Score from result JSON file.")
    parser.add_argument('--result', type=str, required=True, help='Path to the result JSON file.')

    args = parser.parse_args()
    for name, score in calculate_scores(args.result).items():
        print(f"AP ({name}): {score:.4f}")
