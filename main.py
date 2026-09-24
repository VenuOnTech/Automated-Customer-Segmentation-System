import os
import json
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import hdbscan
import umap
from sklearn.preprocessing import StandardScaler

from src.preprocessing.sequence_builder import build_customer_trajectories, pad_sequences
from src.representation.trajectory_transformer import CustomerTrajectoryTransformer
from src.training.contrastive_trainer import InfoNCELoss
from src.evaluation.cluster_evaluation import evaluate_clustering_quality, benchmark_pipeline
from src.feature_engineering.rfm_features import create_rfm
from src.segmentation.kmeans_segmentation import run_kmeans

def run_preprocessing(raw_path="data/raw/Online_Retail.xlsx", parquet_path="data/processed/Online_Retail.parquet"):
    os.makedirs("data/processed", exist_ok=True)
    if not os.path.exists(parquet_path):
        print(f"📥 Converting {raw_path} to optimized Parquet...")
        df = pd.read_excel(raw_path)
        df["StockCode"] = df["StockCode"].astype(str)
        df["InvoiceNo"] = df["InvoiceNo"].astype(str)
        if "Description" in df.columns:
            df["Description"] = df["Description"].astype(str)
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
    
    print(f"📂 Loading dataset from {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)
        
    print("⏳ Building customer sequences...")
    trajectories, item2idx = build_customer_trajectories(df, min_baskets=2)
    
    # Save vocabulary for the API
    os.makedirs("models", exist_ok=True)
    with open("models/item2idx.json", "w") as f:
        json.dump(item2idx, f)
        
    return df, trajectories, item2idx

def run_training(trajectories, vocab_size):
    cids, baskets, deltas, padding_mask = pad_sequences(trajectories)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CustomerTrajectoryTransformer(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = InfoNCELoss(temperature=0.1)
    
    baskets, deltas, padding_mask = baskets.to(device), deltas.to(device), padding_mask.to(device)
    
    print("🧠 Training Trajectory Transformer via InfoNCE...")
    model.train()
    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        mask_v1 = torch.rand(baskets.shape, device=device) > 0.1
        baskets_v1 = baskets * mask_v1.long()
        deltas_v2 = deltas * (1.0 + torch.randn_like(deltas) * 0.05)
        
        z_i, _ = model(baskets_v1, deltas, padding_mask)
        z_j, _ = model(baskets, deltas_v2, padding_mask)
        
        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch [{epoch+1}/{epochs}] | Contrastive Loss: {loss.item():.4f}")
            
    # Save trained PyTorch model
    torch.save(model.state_dict(), "models/transformer.pth")
    return model, cids, baskets, deltas, padding_mask

def run_clustering(model, cids, baskets, deltas, padding_mask, df, trajectories):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        z_final, _ = model(baskets.to(device), deltas.to(device), padding_mask.to(device))
        embeddings = z_final.cpu().numpy()
        
    print("📊 Clustering high-dimensional representations...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    cluster_labels = clusterer.fit_predict(embeddings)
    transformer_metrics = evaluate_clustering_quality(embeddings, cluster_labels)
    
    print("📈 Computing baseline RFM metrics for benchmark...")
    mapping = {"customer_id": "CustomerID", "transaction_date": "InvoiceDate", "quantity": "Quantity", "price": "UnitPrice"}
    rfm_df = create_rfm(df, mapping).fillna(0)
    rfm_df = rfm_df[rfm_df["CustomerID"].isin(cids)]
    rfm_features = rfm_df[["Recency", "Frequency", "Monetary"]].values
    rfm_scaled = StandardScaler().fit_transform(rfm_features)
    _, rfm_labels, _, _ = run_kmeans(rfm_df, config={"clustering": {"n_clusters": 4}})
    rfm_metrics = evaluate_clustering_quality(rfm_scaled, rfm_labels)
    
    benchmark_pipeline(rfm_metrics, transformer_metrics)
    
    print("🎨 Computing 2D UMAP projection...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(embeddings)
    
    results_df = pd.DataFrame({
        "CustomerID": cids,
        "Cluster": cluster_labels,
        "UMAP_1": umap_coords[:, 0],
        "UMAP_2": umap_coords[:, 1]
    })
    results_df["Recent_Items"] = [trajectories[cid]["items"][-1][:5] for cid in cids]
    
    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/transformer_segments.csv", index=False)
    print("\n✅ SYSTEM COMPLETE: Representations saved to outputs/transformer_segments.csv\n")

def run():
    print("\n🚀 STARTING SELF-SUPERVISED CUSTOMER TRAJECTORY PIPELINE\n")
    df, trajectories, item2idx = run_preprocessing()
    model, cids, baskets, deltas, padding_mask = run_training(trajectories, len(item2idx))
    run_clustering(model, cids, baskets, deltas, padding_mask, df, trajectories)

if __name__ == "__main__":
    run()