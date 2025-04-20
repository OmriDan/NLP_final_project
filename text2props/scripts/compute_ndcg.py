import glob
import pickle
import numpy as np
from sklearn.metrics import ndcg_score, accuracy_score

def bucketize_to_10(x):
    """
    Convert float in [0,1] back to one of 10 integer buckets {0..9}.
    Original labels were (diff-1)/9 for diff in {1..10}, so rounding x*9 recovers the bucket.
    """
    return np.rint(np.array(x) * 9).astype(int)

def compute_bucketed_ndcg(y_true, y_pred):
    true_b = bucketize_to_10(y_true)
    pred_b = bucketize_to_10(y_pred)
    # ndcg_score takes list-of-queries, so we wrap each in a list
    return ndcg_score([true_b], [pred_b])

def main():
    # 1) Load known traits
    with open("../data/processed/known_latent_traits_NovaSky.pickle", "rb") as f:
        known = pickle.load(f)
    # key is the DIFFICULTY constant, e.g. "difficulty"
    true_map = known["difficulty"]

    # 2) Find all predicted files
    pred_files = glob.glob("../data/processed/predicted_latent_traits_NovaSky_*.pickle")
    if not pred_files:
        print("No predicted_latent_traits files found.")
        return

    # 3) For each experiment, compute nDCG
    for path in sorted(pred_files):
        with open(path, "rb") as f:
            preds = pickle.load(f)

        # align by Q_ID
        q_ids = list(preds.keys())
        y_pred = [preds[q]       for q in q_ids]
        y_true = [true_map[q]    for q in q_ids]

        score = compute_bucketed_ndcg(y_true, y_pred)
        exp_name = path.replace("predicted_latent_traits_NovaSky_", "").replace(".pickle", "")
        print(f"{exp_name:30s} →  bucketed nDCG = {score:.4f}")
        
        true_b = bucketize_to_10(y_true)
        pred_b = bucketize_to_10(y_pred)
        acc = accuracy_score(true_b, pred_b)
        print(f"{exp_name:30s} →  accuracy = {acc:.4f}, bucketed nDCG = {score:.4f}")

if __name__ == "__main__":
    main()