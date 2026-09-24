import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
from src.preprocessing.sequence_builder import build_customer_trajectories
from src.representation.trajectory_transformer import CustomerTrajectoryTransformer
from src.training.contrastive_trainer import InfoNCELoss
from src.evaluation.cluster_evaluation import evaluate_clustering_quality, benchmark_pipeline
from src.feature_engineering.rfm_features import create_rfm
from src.segmentation.kmeans_segmentation import run_kmeans

# UMAP for dashboard visualization
import umap

def pad_sequences(trajectories, max_seq_len=20, max_basket_size=10):
    """Formats raw lists into padded PyTorch tensors."""
    customer_ids = list(trajectories.keys())
    N = len(customer_ids)
    
    baskets = torch.zeros((N, max_seq_len, max_basket_size), dtype=torch.long)
    deltas = torch.zeros((N, max_seq_len), dtype=torch.float32)
    padding_mask = torch.ones((N, max_seq_len), dtype=torch.bool)
    
    for i, cid in enumerate(customer_ids):
        user_data = trajectories[cid]
        items = user_data["items"][-max_seq_len:]
        time_d = user_data["time_deltas"][-max_seq_len:]
        
        seq_len = len(items)
        padding_mask[i, :seq_len] = False
        deltas[i, :seq_len] = torch.tensor(time_d)
        
        for j, basket in enumerate(items):
            b_len = min(len(basket), max_basket_size)
            baskets[i, j, :b_len] = torch.tensor(basket[:b_len])
            
    return customer_ids, baskets, deltas, padding_mask

def run():
    print("\n🚀 STARTING SELF-SUPERVISED CUSTOMER TRAJECTORY PIPELINE\n")
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Ingest & Convert to Parquet
    raw_path = "data/raw/Online_Retail.xlsx"
    parquet_path = "data/processed/Online_Retail.parquet"
    os.makedirs("data/processed", exist_ok=True)
    
    if not os.path.exists(parquet_path):
        print(f"📥 Converting {raw_path} to optimized Parquet...")
        df = pd.read_excel(raw_path)
        df.to_parquet(parquet_path, engine="fastparquet")
    
    print(f"📂 Loading dataset from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Optional Lite Mode for CI/Testing
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)
        
    # 2. Extract Trajectories
    print("⏳ Building customer sequences and temporal intervals...")
    trajectories, item2idx = build_customer_trajectories(df, min_baskets=2)
    vocab_size = len(item2idx)
    print(f"✅ Extracted {len(trajectories)} viable customer trajectories. Vocab size: {vocab_size}")
    
    cids, baskets, deltas, padding_mask = pad_sequences(trajectories)
    
    # 3. Model Initialization (z_i \in \mathbb{R}^{64})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomerTrajectoryTransformer(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = InfoNCELoss(temperature=0.1)
    
    baskets = baskets.to(device)
    deltas = deltas.to(device)
    padding_mask = padding_mask.to(device)
    
    # 4. Contrastive Training Loop (Self-Supervised)
    print("🧠 Training Trajectory Transformer via InfoNCE...")
    model.train()
    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Temporal Augmentation: Create two sub-sequences (views) for contrastive learning
        # View 1: Drop random items in baskets
        mask_v1 = torch.rand(baskets.shape, device=device) > 0.1
        baskets_v1 = baskets * mask_v1.long()
        
        # View 2: Shift time deltas slightly
        deltas_v2 = deltas * (1.0 + torch.randn_like(deltas) * 0.05)
        
        z_i, _ = model(baskets_v1, deltas, padding_mask)
        z_j, _ = model(baskets, deltas_v2, padding_mask)
        
        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch [{epoch+1}/{epochs}] | Contrastive Loss: {loss.item():.4f}")

    # 5. Extract Final Representations
    model.eval()
    with torch.no_grad():
        z_final, _ = model(baskets, deltas, padding_mask)
        embeddings = z_final.cpu().numpy()
        
    # 6. Clustering (HDBSCAN on \mathbb{R}^{64})
    print("📊 Clustering high-dimensional representations...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    cluster_labels = clusterer.fit_predict(embeddings)
    
    transformer_metrics = evaluate_clustering_quality(embeddings, cluster_labels)
    
    # 7. Baseline Comparison (RFM + KMeans)
    print("📈 Computing baseline RFM metrics for benchmark...")
    mapping = {"customer_id": "CustomerID", "transaction_date": "InvoiceDate", "quantity": "Quantity", "price": "UnitPrice"}
    rfm_df = create_rfm(df, mapping).fillna(0)
    rfm_df = rfm_df[rfm_df["CustomerID"].isin(cids)]
    rfm_features = rfm_df[["Recency", "Frequency", "Monetary"]].values
    rfm_scaled = StandardScaler().fit_transform(rfm_features)
    _, rfm_labels = run_kmeans(rfm_df, config={"clustering": {"n_clusters": 4}})[:2]
    
    rfm_metrics = evaluate_clustering_quality(rfm_scaled, rfm_labels)
    benchmark_pipeline(rfm_metrics, transformer_metrics)
    
    # 8. Dimensionality Reduction for Dashboard (UMAP)
    print("🎨 Computing 2D UMAP projection for Dashboard...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(embeddings)
    
    # 9. Save Results
    results_df = pd.DataFrame({
        "CustomerID": cids,
        "Cluster": cluster_labels,
        "UMAP_1": umap_coords[:, 0],
        "UMAP_2": umap_coords[:, 1]
    })
    
    # Attach raw characteristic sequences for the dashboard
    results_df["Recent_Items"] = [trajectories[cid]["items"][-1][:5] for cid in cids]
    results_df.to_csv("outputs/transformer_segments.csv", index=False)
    
    print("\n✅ SYSTEM COMPLETE: Representations saved to outputs/transformer_segments.csv\n")

if __name__ == "__main__":
    run()