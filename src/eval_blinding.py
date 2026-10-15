import json
import random
from typing import List, Dict, Any
import numpy as np

def generate_blinded_review_bundle(
    model_runs: List[Dict[str, Any]], 
    eval_jsonl_path: str = "data/blinded_eval_set.jsonl",
    researcher_private_key_path: str = "researcher_only/lookup_key.json",
    seed: int = 42
):
    import os
    os.makedirs(os.path.dirname(eval_jsonl_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(researcher_private_key_path) or ".", exist_ok=True)

    rng = random.Random(seed)
    blinded_records = []
    lookup_key = {}

    indices = list(range(len(model_runs)))
    rng.shuffle(indices)

    for rank, idx in enumerate(indices):
        item = model_runs[idx]
        token = f"EVAL_{rank+1:04d}"
        lookup_key[token] = {
            "variant": item["system_variant"],
            "prompt_id": item["prompt_id"]
        }
        blinded_records.append({
            "eval_token": token,
            "prompt_hausa": item["prompt_hausa"],
            "grounding_context": item.get("grounding_context", "NONE_PROVIDED"),
            "system_response_hausa": item["final_output"],
            "groundedness_score": None,  # [1-5]
            "hausa_quality_score": None  # [1-5]
        })

    with open(eval_jsonl_path, "w", encoding="utf-8") as f:
        for r in blinded_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(researcher_private_key_path, "w", encoding="utf-8") as f:
        json.dump(lookup_key, f, indent=2)

    print(f"Generated {len(blinded_records)} blinded tasks. Key secured separately at {researcher_private_key_path}.")


def compute_quadratic_weighted_kappa(rater_a: List[int], rater_b: List[int], min_rating: int = 1, max_rating: int = 5) -> float:
    """
    Computes Cohen's Quadratic Weighted Kappa (QWK) on ordinal clinical scoring axes.
    """
    assert len(rater_a) == len(rater_b), "Rater length mismatch"
    categories = list(range(min_rating, max_rating + 1))
    num_cats = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    O = np.zeros((num_cats, num_cats), dtype=float)
    for a, b in zip(rater_a, rater_b):
        O[cat_to_idx[a], cat_to_idx[b]] += 1.0

    W = np.zeros((num_cats, num_cats), dtype=float)
    for i in range(num_cats):
        for j in range(num_cats):
            W[i, j] = ((i - j) ** 2) / ((num_cats - 1) ** 2)

    n = len(rater_a)
    hist_a = np.sum(O, axis=1)
    hist_b = np.sum(O, axis=0)
    E = np.outer(hist_a, hist_b) / n

    numerator = np.sum(W * O)
    denominator = np.sum(W * E)

    if denominator == 0:
        return 1.0
    return 1.0 - (numerator / denominator)