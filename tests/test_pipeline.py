import sys
import os
import pandas as pd
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

def test_imports():
    try:
        from src.preprocessing.sequence_builder import build_customer_trajectories, pad_sequences
        from src.representation.trajectory_transformer import CustomerTrajectoryTransformer
        from src.training.contrastive_trainer import InfoNCELoss
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_sequence_builder():
    try:
        from src.preprocessing.sequence_builder import build_customer_trajectories
        
        # Mock transaction data
        df = pd.DataFrame({
            'CustomerID': [1, 1, 1, 2, 2, 2],
            'InvoiceNo': ['100', '100', '101', '200', '201', '202'],
            'InvoiceDate': ['2023-01-01', '2023-01-01', '2023-01-05', '2023-01-01', '2023-01-10', '2023-01-20'],
            'StockCode': ['A', 'B', 'A', 'C', 'C', 'D'],
            'Quantity': [1, 2, 1, 1, 1, 3],
            'UnitPrice': [10.0, 20.0, 10.0, 5.0, 5.0, 15.0]
        })
        
        trajectories, item2idx = build_customer_trajectories(df, min_baskets=2)
        
        # Assertions
        assert 1 in trajectories
        assert 2 in trajectories
        assert len(item2idx) == 4  # A, B, C, D
        assert trajectories[1]["time_deltas"] == [0.0, 4.0] # 4 days between Jan 1 and Jan 5
        print("✅ Sequence builder test passed")
        return True
    except Exception as e:
        print(f"❌ Sequence builder test failed: {e}")
        return False

def test_tensor_padding():
    try:
        from src.preprocessing.sequence_builder import pad_sequences
        
        mock_trajectories = {
            99: {
                "items": [[1, 2], [3]],
                "time_deltas": [0.0, 5.0],
                "basket_values": [30.0, 10.0]
            }
        }
        
        cids, baskets, deltas, padding_mask = pad_sequences(mock_trajectories, max_seq_len=5, max_basket_size=3)
        
        assert list(baskets.shape) == [1, 5, 3] # (N, seq_len, basket_size)
        assert list(deltas.shape) == [1, 5]
        assert not padding_mask[0, 0] # First position should be unmasked (False)
        assert padding_mask[0, 4]     # Last position should be masked (True)
        print("✅ Tensor padding test passed")
        return True
    except Exception as e:
        print(f"❌ Tensor padding test failed: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧪 RUNNING TRAJECTORY PIPELINE TESTS")
    print("="*50 + "\n")
    
    results = [
        test_imports(),
        test_sequence_builder(),
        test_tensor_padding()
    ]
    
    print("\n" + "="*50)
    passed = sum(1 for r in results if r)
    if all(results):
        print(f"✅ ALL TESTS PASSED ({passed}/{len(results)})")
        print("="*50 + "\n")
        sys.exit(0)
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{len(results)} passed)")
        print("="*50 + "\n")
        sys.exit(1)