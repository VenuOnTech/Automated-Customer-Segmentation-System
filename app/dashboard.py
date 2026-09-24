import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(
    page_title="AI Trajectory Segmentation",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Self-Supervised Customer Trajectory Segmentation")
st.markdown("### 🚀 Contrastive Representation Learning on Sequential Retail Data")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
local_path = os.path.join(BASE_DIR, "outputs", "transformer_segments.csv")

@st.cache_data
def load_data():
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    return None

df = load_data()

if df is not None and len(df) > 0:
    st.success("✅ Deep Sequence Representations Loaded")
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Modeled Customers", f"{len(df):,}")
    
    # HDBSCAN maps noise to -1
    valid_clusters = df[df["Cluster"] != -1]["Cluster"].nunique()
    col2.metric("🎯 Valid Latent Clusters", valid_clusters)
    col3.metric("⚠️ Noise Entities", (df["Cluster"] == -1).sum())
    
    st.divider()
    
    # 2D UMAP Projection of \mathbb{R}^{64} representations
    st.subheader("🌌 UMAP Projection of Trajectory Latent Space")
    df["Cluster_Label"] = df["Cluster"].astype(str)
    
    fig = px.scatter(
        df,
        x="UMAP_1",
        y="UMAP_2",
        color="Cluster_Label",
        hover_data=["CustomerID", "Recent_Items"],
        title="Customer Embeddings (Contrastive Sequence Model)",
        color_discrete_sequence=px.colors.qualitative.G10
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    tab1, tab2 = st.tabs(["📋 Data Lineage", "🔍 Sequence Insights"])
    
    with tab1:
        st.dataframe(df.drop(columns=["Cluster_Label"]), use_container_width=True)
        
    with tab2:
        st.subheader("🧬 Characteristic Item Sequences per Cluster")
        for cluster_id in sorted(df["Cluster"].unique()):
            if cluster_id == -1:
                continue
            c_data = df[df["Cluster"] == cluster_id]
            st.markdown(f"**Cluster {cluster_id}** ({len(c_data)} customers)")
            # Displaying typical transition sequences [source: 2]
            sample_seqs = c_data.sample(min(3, len(c_data)))["Recent_Items"].tolist()
            for seq in sample_seqs:
                st.code(f"Item Sequence: {seq}", language="text")
else:
    st.error("❌ No trajectory representations available. Run main.py first.")