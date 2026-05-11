import json
import argparse

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


def calculate_scores(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        y_label = []
        y_prob = []
        uncertainty_estimators = data.pop(0)[3:]
        uncertainty_scores_sorted = ([], [])
        for sample in data:
            prediction, prediction_probability, confidence_probability = sample[0:3]
            uncertainty_scores = sample[3:]

            y_label.append(prediction)
            y_prob.append(0.5 * prediction_probability + 0.5 * confidence_probability)

            uncertainty_scores_sorted[1 if confidence_probability > 0.5 else 0].append(uncertainty_scores)

        p1, r1, _ = precision_recall_curve(y_label, y_prob)
        ap_score = average_precision_score(y_label, y_prob)

        mean_uncertainty_sure = np.mean(uncertainty_scores_sorted[1], axis=0)
        mean_uncertainty_unsure = np.mean(uncertainty_scores_sorted[0], axis=0)
        mean_uncertainty_overall = np.mean(np.vstack(uncertainty_scores_sorted), axis=0)

        uncertainty_scores = {
            name: (uncertainty_score_sure, uncertainty_score_unsure, uncertainty_score_overall)
            for name, uncertainty_score_sure, uncertainty_score_unsure, uncertainty_score_overall
            in zip(uncertainty_estimators, mean_uncertainty_sure, mean_uncertainty_unsure, mean_uncertainty_overall)
        }

        refusal_rate = len(uncertainty_scores_sorted[0]) / len(y_label)

        return ap_score, uncertainty_scores, refusal_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Average Precision Score from result JSON file.")
    parser.add_argument('--result', type=str, required=True, help='Path to the result JSON file.')

    args = parser.parse_args()
    ap_score, uncertainty_scores, refusal_rate = calculate_scores(args.result)
    print(f"Average Precision Score: {ap_score}")
    for uncertainty_score_name, (uncertainty_score_sure, uncertainty_score_unsure, uncertainty_score_overall) in uncertainty_scores.items():
        print(f"{uncertainty_score_name} (U/S/O/U-S): "
              f"{uncertainty_score_sure:.4f} {uncertainty_score_unsure:.4f} {uncertainty_score_overall:.4f} "
              f"{uncertainty_score_unsure - uncertainty_score_sure:.4f}")
    print(f"Refusal Rate: {refusal_rate}")
