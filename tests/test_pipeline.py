"""
Unit tests for the pipeline
"""
import sys
import os
import pandas as pd

# Add parent directory to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


# ==============================
# TEST IMPORTS
# ==============================
def test_imports():
    try:
        from src.data_ingestion.load_data import load_data
        from src.preprocessing.data_cleaning import clean_data
        from src.preprocessing.sequence_builder import build_customer_trajectories
        from src.representation.trajectory_transformer import CustomerTrajectoryTransformer
        from src.training.contrastive_trainer import InfoNCELoss
        from src.evaluation.cluster_evaluation import evaluate_clustering_quality
        from src.feature_engineering.rfm_features import create_rfm
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


# ==============================
# SCHEMA TEST
# ==============================
def test_schema_detection():
    try:
        from src.data_ingestion.schema_detection import detect_columns

        df = pd.DataFrame({
            'CustomerID': [1, 2, 3],
            'InvoiceDate': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'Quantity': [1, 2, 3],
            'UnitPrice': [10.0, 20.0, 30.0]
        })

        mapping = detect_columns(df)

        assert mapping['customer_id'] == 'CustomerID'
        assert mapping['transaction_date'] == 'InvoiceDate'
        assert mapping['quantity'] == 'Quantity'
        assert mapping['price'] == 'UnitPrice'

        print("✅ Schema detection test passed")
        return True

    except Exception as e:
        print(f"❌ Schema detection test failed: {e}")
        return False


# ==============================
# SCHEMA ERROR TEST
# ==============================
def test_schema_detection_missing_columns():
    try:
        from src.data_ingestion.schema_detection import detect_columns

        df = pd.DataFrame({
            'CustomerID': [1, 2, 3],
            'InvoiceDate': ['2021-01-01', '2021-01-02', '2021-01-03']
        })

        try:
            detect_columns(df)
            print("❌ Should have raised ValueError")
            return False

        except ValueError as e:
            if "Missing required columns" in str(e):
                print("✅ Schema detection error handling test passed")
                return True
            else:
                print(f"❌ Wrong error message: {e}")
                return False

    except Exception as e:
        print(f"❌ Schema detection error test failed: {e}")
        return False


# ==============================
# DATA CLEANING TEST
# ==============================
def test_data_cleaning():
    try:
        from src.preprocessing.data_cleaning import clean_data

        df = pd.DataFrame({
            'CustomerID': [1, 2, 3, 4, None],
            'InvoiceDate': ['2021-01-01'] * 5,
            'Quantity': [1, 2, 3, -1, 5],
            'UnitPrice': [10.0, 20.0, 30.0, 40.0, -50.0]
        })

        mapping = {
            'customer_id': 'CustomerID',
            'transaction_date': 'InvoiceDate',
            'quantity': 'Quantity',
            'price': 'UnitPrice'
        }

        df_clean = clean_data(df, mapping)

        assert len(df_clean) == 3
        assert df_clean['Quantity'].min() > 0
        assert df_clean['UnitPrice'].min() > 0
        assert df_clean['CustomerID'].isnull().sum() == 0

        print("✅ Data cleaning test passed")
        return True

    except Exception as e:
        print(f"❌ Data cleaning test failed: {e}")
        return False


# ==============================
# RFM TEST
# ==============================
def test_rfm_creation():
    try:
        from src.feature_engineering.rfm_features import create_rfm

        df = pd.DataFrame({
            'CustomerID': [1, 1, 2, 2, 3],
            'InvoiceDate': ['2021-01-01', '2021-01-05', '2021-01-02', '2021-01-10', '2021-01-03'],
            'Quantity': [1, 2, 3, 1, 2],
            'UnitPrice': [10.0, 20.0, 30.0, 40.0, 50.0]
        })

        mapping = {
            'customer_id': 'CustomerID',
            'transaction_date': 'InvoiceDate',
            'quantity': 'Quantity',
            'price': 'UnitPrice'
        }

        rfm = create_rfm(df, mapping)

        assert 'Recency' in rfm.columns
        assert 'Frequency' in rfm.columns
        assert 'Monetary' in rfm.columns
        assert len(rfm) == 3

        print("✅ RFM creation test passed")
        return True

    except Exception as e:
        print(f"❌ RFM creation test failed: {e}")
        return False


# ==============================
# RUN ALL TESTS
# ==============================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧪 RUNNING PIPELINE TESTS")
    print("="*50 + "\n")

    results = [
        test_imports(),
        test_schema_detection(),
        test_schema_detection_missing_columns(),
        test_data_cleaning(),
        test_rfm_creation()
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