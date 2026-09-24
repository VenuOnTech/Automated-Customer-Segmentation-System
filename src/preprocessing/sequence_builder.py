import pandas as pd
import numpy as np

def build_customer_trajectories(df: pd.DataFrame, min_baskets: int = 3):
    """
    Transforms transaction logs into chronological baskets per customer.
    Filters cancellations (InvoiceNo starting with 'C').
    """
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")].copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.sort_values(by=["CustomerID", "InvoiceDate"])

    # Map StockCodes to continuous integer tokens
    unique_items = df["StockCode"].unique()
    item2idx = {code: idx + 1 for idx, code in enumerate(unique_items)} # 0 reserved for padding
    df["ItemToken"] = df["StockCode"].map(item2idx)

    trajectories = {}
    grouped = df.groupby(["CustomerID", "InvoiceNo", "InvoiceDate"])

    baskets_df = grouped.agg({
        "ItemToken": list,
        "Quantity": list,
        "UnitPrice": lambda x: (x * df.loc[x.index, "Quantity"]).sum() # Basket Monetary Value
    }).reset_index()

    baskets_df = baskets_df.sort_values(by=["CustomerID", "InvoiceDate"])

    for customer_id, user_baskets in baskets_df.groupby("CustomerID"):
        if len(user_baskets) < min_baskets:
            continue

        timestamps = user_baskets["InvoiceDate"].values
        deltas = np.zeros(len(timestamps), dtype=np.float32)
        # Calculate inter-purchase intervals in days
        deltas[1:] = (timestamps[1:] - timestamps[:-1]) / np.timedelta64(1, "D")

        trajectories[customer_id] = {
            "items": user_baskets["ItemToken"].tolist(),
            "time_deltas": deltas.tolist(),
            "basket_values": user_baskets["UnitPrice"].tolist()
        }

    return trajectories, item2idx