"""Non-LLM baseline: 1-nearest-neighbour classification over TF-IDF vectors.

Uses the same stratified 4-fold cross-validation as run_all.py (n_splits=4,
shuffle, random_state=42), so the scores are directly comparable with the
prompting strategies. Conceptually this mirrors dynamic few-shot retrieval
with k=1: each test post is assigned the label of its most similar training
post.

Per-fold result CSVs are written to {results}/final_run/knn_baseline/ in the
standard result format (id, text, true_label, predicted_label; no reply
column since there is no model output). Per-fold and mean ± std scores are
printed to stdout.

Run from src/; paths are relative to this directory (unlike config.yaml,
which uses the container-internal absolute paths /data and /results).
"""

import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

DATA_PATH = "../data"
RESULTS_PATH = "../results"
N_SPLITS = 4


def get_data_splits(fold, df):
    """Split df into (train, test) using the row indices of one CV fold."""
    return df.loc[fold[0]], df.loc[fold[1]]


def main():
    df = pd.read_csv(os.path.join(DATA_PATH, "def_train.csv"), sep=";")
    # The DEF column holds the labels as strings — normalize to bool
    df["DEF"] = df["DEF"].map(lambda x: str(x).strip().upper() == "TRUE")

    # Same CV setup as run_all.py so folds are identical across experiments
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = {
        i: [train_index, test_index]
        for i, (train_index, test_index) in
        enumerate(skf.split(df["description"], df["DEF"]))
    }

    output_dir = os.path.join(RESULTS_PATH, "final_run", "knn_baseline")
    os.makedirs(output_dir, exist_ok=True)

    scores = []

    for fold_idx in range(N_SPLITS):
        train, test = get_data_splits(folds[fold_idx], df)

        # TF-IDF vocabulary is fitted on the train split only
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(train["description"])
        X_test = vectorizer.transform(test["description"])

        # k=1: predict the label of the single most similar training post
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X_train, train["DEF"])
        y_pred = clf.predict(X_test)

        output_path = os.path.join(output_dir, f"knn-baseline_fold-{fold_idx}.csv")
        df_out = pd.DataFrame({
            "id": test["id"],
            "text": test["description"],
            "true_label": test["DEF"],
            "predicted_label": y_pred,
        })
        df_out.to_csv(output_path, index=False)

        print(f"Fold {fold_idx} done → {output_path}")

        # Mirror evaluate.py, which excludes abstentions (NaN predictions);
        # the KNN never abstains, so the mask keeps every row here
        mask = pd.notna(df_out["predicted_label"])
        y_true = df_out.loc[mask, "true_label"].astype(bool)
        y_pred = df_out.loc[mask, "predicted_label"].astype(bool)

        # Precision/recall refer to the positive (prosecutable) class;
        # F1 Macro is the primary comparison metric of the project
        fold_scores = {
            "fold": fold_idx,
            "n": mask.sum(),
            "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred), 4),
            "recall": round(recall_score(y_true, y_pred), 4),
        }
        scores.append(fold_scores)
        print(f"Fold {fold_idx}: F1={fold_scores['f1_macro']:.4f}  "
            f"Acc={fold_scores['accuracy']:.4f}  "
            f"Prec={fold_scores['precision']:.4f}  "
            f"Rec={fold_scores['recall']:.4f}")

    score_df = pd.DataFrame(scores)
    mean = score_df.drop(columns="fold").mean()
    std = score_df.drop(columns="fold").std()

    print(f"\nMean ± std across {N_SPLITS} folds:")
    for col in ["f1_macro", "accuracy", "precision", "recall"]:
        print(f"  {col}: {mean[col]:.4f} ± {std[col]:.4f}")


if __name__ == "__main__":
    main()
