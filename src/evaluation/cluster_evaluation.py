import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

def evaluate_clustering_quality(X, labels):
    """
    Computes internal cluster validation metrics on the representation space \mathbb{R}^d.
    """
    try:
        if len(set(labels)) < 2:
            return {"silhouette": -1.0, "davies_bouldin": -1.0}
        
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        return {"silhouette": float(sil), "davies_bouldin": float(db)}
    except Exception as e:
        print(f"⚠️ Clustering evaluation failed: {e}")
        return {"silhouette": -1.0, "davies_bouldin": -1.0}

def evaluate_temporal_stability(labels_q1, labels_q2):
    """
    Computes the Adjusted Rand Index (ARI) to measure if clusters 
    remain stable across temporal splits.
    """
    try:
        ari = adjusted_rand_score(labels_q1, labels_q2)
        return {"ari": float(ari)}
    except Exception:
        return {"ari": -1.0}

def evaluate_next_basket(predictions, targets, k=10):
    """
    Computes Hit-Ratio@10 and NDCG@10 for next-basket prediction.
    """
    hits = 0
    ndcg = 0.0
    
    for pred_basket, true_item in zip(predictions, targets):
        if true_item in pred_basket[:k]:
            hits += 1
            rank = pred_basket[:k].index(true_item)
            ndcg += 1.0 / np.log2(rank + 2)
            
    n = len(targets) if len(targets) > 0 else 1
    return {
        "hr_at_10": float(hits / n), 
        "ndcg_at_10": float(ndcg / n)
    }

def benchmark_pipeline(rfm_metrics, transformer_metrics):
    print("\n" + "="*50)
    print("🔬 RESEARCH BENCHMARK: Baseline vs Proposed")
    print("="*50)
    print(f"Metrics             | Baseline (RFM) | Proposed (Transformer)")
    print("-" * 50)
    print(f"Silhouette Score    | {rfm_metrics.get('silhouette', 0):.4f}         | {transformer_metrics.get('silhouette', 0):.4f}")
    print(f"Davies-Bouldin      | {rfm_metrics.get('davies_bouldin', 0):.4f}         | {transformer_metrics.get('davies_bouldin', 0):.4f}")
    print("="*50 + "\n")