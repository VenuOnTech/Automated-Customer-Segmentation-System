from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import os
import numpy as np
from src.representation.trajectory_transformer import CustomerTrajectoryTransformer

app = FastAPI(title="Customer Trajectory Embedding API")

# Global variables for model and vocab
model = None
item2idx = {}

class Basket(BaseModel):
    items: list[str]
    time_since_last_days: float

class TrajectoryRequest(BaseModel):
    baskets: list[Basket]

@app.on_event("startup")
def load_assets():
    global model, item2idx
    vocab_path = "models/item2idx.json"
    model_path = "models/transformer.pth"
    
    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        print("⚠️ Model or vocabulary not found. Run main.py first to train.")
        return
        
    with open(vocab_path, "r") as f:
        item2idx = json.load(f)
        
    vocab_size = len(item2idx)
    model = CustomerTrajectoryTransformer(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("✅ Model and vocabulary loaded successfully.")

@app.get("/")
def home():
    return {"status": "API running", "model_loaded": model is not None}

@app.post("/embed")
def get_embedding(req: TrajectoryRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
        
    # Formatting input to match training constraints (max 20 baskets, max 10 items)
    seq_len = min(len(req.baskets), 20)
    baskets_tensor = torch.zeros((1, 20, 10), dtype=torch.long)
    deltas_tensor = torch.zeros((1, 20), dtype=torch.float32)
    padding_mask = torch.ones((1, 20), dtype=torch.bool)
    
    for i in range(seq_len):
        basket = req.baskets[i]
        padding_mask[0, i] = False
        deltas_tensor[0, i] = basket.time_since_last_days
        
        # Tokenize items using training vocabulary (0 for unknown)
        tokens = [item2idx.get(str(item), 0) for item in basket.items[:10]]
        for j, token in enumerate(tokens):
            baskets_tensor[0, i, j] = token
            
    with torch.no_grad():
        z_i, _ = model(baskets_tensor, deltas_tensor, padding_mask)
        
    return {
        "embedding_vector": z_i[0].tolist(),
        "dimensions": 64
    }