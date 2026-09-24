import pandas as pd
import numpy as np
import torch

def build_customer_trajectories(df: pd.DataFrame, min_baskets: int = 3):
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")].copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.sort_values(by=["CustomerID", "InvoiceDate"])
    
    unique_items = df["StockCode"].unique()
    item2idx = {code: idx + 1 for idx, code in enumerate(unique_items)} 
    df["ItemToken"] = df["StockCode"].map(item2idx)
    
    trajectories = {}
    grouped = df.groupby(["CustomerID", "InvoiceNo", "InvoiceDate"])
    baskets_df = grouped.agg({
        "ItemToken": list,
        "Quantity": list,
        "UnitPrice": lambda x: (x * df.loc[x.index, "Quantity"]).sum()
    }).reset_index()
    
    baskets_df = baskets_df.sort_values(by=["CustomerID", "InvoiceDate"])
    for customer_id, user_baskets in baskets_df.groupby("CustomerID"):
        if len(user_baskets) < min_baskets:
            continue
        timestamps = user_baskets["InvoiceDate"].values
        deltas = np.zeros(len(timestamps), dtype=np.float32)
        deltas[1:] = (timestamps[1:] - timestamps[:-1]) / np.timedelta64(1, "D")
        
        trajectories[customer_id] = {
            "items": user_baskets["ItemToken"].tolist(),
            "time_deltas": deltas.tolist(),
            "basket_values": user_baskets["UnitPrice"].tolist()
        }
    return trajectories, item2idx

def pad_sequences(trajectories, max_seq_len=20, max_basket_size=10):
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