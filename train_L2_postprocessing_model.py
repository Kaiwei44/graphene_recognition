import json
import os

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from maskterial.utils.argparser import parse_postprocessing_args
from maskterial.utils.loader_functions import load_and_partition_data

if __name__ == "__main__":
    args = parse_postprocessing_args()
    DATA_DIR = args.data_dir
    SAVE_DIR = args.save_dir
    CFG_PATH = args.config
    TRAIN_SEED = args.train_seed

    meta_out = os.path.join(SAVE_DIR, "metadata.json")
    model_out = os.path.join(SAVE_DIR, "model.joblib")

    with open(CFG_PATH, "r") as f:
        CONFIG = json.load(f)
        fit_config = CONFIG["fit_config"]

    os.makedirs(SAVE_DIR, exist_ok=True)

    np.random.seed(TRAIN_SEED)
    X_train, y_train, X_test, y_test = load_and_partition_data(DATA_DIR)

    np.random.seed(TRAIN_SEED)
    classifier = LogisticRegression(
        **fit_config,
        random_state=TRAIN_SEED,
    )

    classifier.fit(X_train, y_train)
    dump(classifier, model_out)

    # predict on the test set
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    with open(meta_out, "w") as f:
        json.dump(
            {
                "metrics": {
                    "accuracy": accuracy,
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                },
                "fit_config": fit_config,
                "random_state": TRAIN_SEED,
            },
            f,
        )

    print(
        f"Accuracy: {accuracy:.2%} | Recall: {recall:.2%} | Precision: {precision:.2%} | F1: {f1:.2%}"
    )
