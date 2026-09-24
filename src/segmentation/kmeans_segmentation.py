from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

def run_kmeans(rfm, config):
    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_clusters = config.get("clustering", {}).get("n_clusters", 4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    rfm["Cluster"] = labels
    
    try:
        sil_score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else None
    except:
        sil_score = None
        
    metrics = {"n_clusters": int(len(set(labels))), "silhouette_score": sil_score}
    
    return rfm, labels, scaler, metrics