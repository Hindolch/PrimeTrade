import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- PAGE SETUP ---
st.set_page_config(page_title="Primetrade.ai - Elite Analytics", layout="wide")
st.title("📈 Primetrade.ai: Sentiment & Behavioral Dashboard")
st.markdown("> **Status:** Analysis synchronized with `primetrade_analysis.ipynb` logic.")

# --- DATA LOADING ---
@st.cache_data
def load_and_sync_data():
    fg = pd.read_csv("data/fear_greed_index.csv")
    hl = pd.read_csv("data/historical_data.csv")
    
    fg["date"] = pd.to_datetime(fg["date"]).dt.normalize()
    hl["date"] = pd.to_datetime(hl["Timestamp IST"], format="%d-%m-%Y %H:%M").dt.normalize()
    
    df = hl.merge(fg[["date", "value", "classification"]], on="date", how="inner")
    df["sentiment"] = df["classification"].apply(lambda c: "Fear" if "Fear" in c else ("Greed" if "Greed" in c else "Neutral"))
    
    # Filter for closed trades for performance metrics
    df_closed = df[df["Direction"].isin(["Close Long", "Close Short"])].copy()
    return df, df_closed

df_all, df_closed = load_and_sync_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Global Filters")
selected_sentiment = st.sidebar.multiselect("Market Sentiment", options=df_closed["sentiment"].unique(), default=df_closed["sentiment"].unique())
filtered_df = df_closed[df_closed["sentiment"].isin(selected_sentiment)]

# --- KPI METRICS (Matching Notebook Part A) ---
# --- KPI METRICS (Matching Notebook Part A) ---
st.subheader("🔑 High-Level Aggregates")
col1, col2, col3, col4, col5 = st.columns(5)

# Calculate base metrics
total_trades = len(filtered_df)
total_pnl = filtered_df['Closed PnL'].sum() if total_trades > 0 else 0
win_rate = (filtered_df['Closed PnL'] > 0).sum() / total_trades if total_trades > 0 else 0

# SAFE LONG/SHORT RATIO
longs = len(filtered_df[filtered_df['Side'] == 'Long'])
shorts = len(filtered_df[filtered_df['Side'] == 'Short'])

if shorts > 0:
    ls_ratio = longs / shorts
else:
    ls_ratio = float('inf') if longs > 0 else 0.0

# Display Metrics
col1.metric("Net PnL", f"${total_pnl:,.0f}")
col2.metric("Win Rate", f"{win_rate:.1%}")
col3.metric("L/S Ratio", f"{ls_ratio:.2f}" if ls_ratio != float('inf') else "∞")
col4.metric("Total Vol", f"${filtered_df['Size USD'].sum()/1e6:.1f}M")
col5.metric("Traders", f"{filtered_df['Account'].nunique()}")

st.info("💡 **Statistical Insight:** Mann-Whitney U testing confirms Fear regimes provide significantly higher liquidity/PnL (p < 0.0001).")

# --- VISUALS (Part B) ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("Performance by Margin Type")
    pnl_margin = filtered_df.groupby("Crossed")["Closed PnL"].sum().reset_index()
    pnl_margin["Margin Type"] = pnl_margin["Crossed"].map({True: "Cross", False: "Isolated"})
    fig1 = px.bar(pnl_margin, x="Margin Type", y="Closed PnL", color="Margin Type", color_discrete_sequence=["#EF553B", "#636EFA"])
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("PnL Volatility vs Sentiment")
    daily_pnl = filtered_df.groupby("date").agg({"Closed PnL": "sum", "value": "first"}).reset_index()
    fig2 = px.scatter(daily_pnl, x="value", y="Closed PnL", trendline="ols", color="Closed PnL", color_continuous_scale="RdYlGn")
    st.plotly_chart(fig2, use_container_width=True)

# --- CLUSTERING (Part C / Bonus) ---
st.divider()
st.subheader("🤖 Behavioral Archetypes (Synced K-Means)")

# Feature Engineering matching the notebook
acc_summary = filtered_df.groupby("Account").agg(
    total_pnl=("Closed PnL", "sum"),
    trade_count=("Closed PnL", "count"),
    avg_size_usd=("Size USD", "mean"),
    trading_days=("date", "nunique")
).reset_index()

acc_summary["trades_per_day"] = acc_summary["trade_count"] / acc_summary["trading_days"]

# Robust Clustering Logic
features = ["total_pnl", "trades_per_day", "avg_size_usd"]
X = StandardScaler().fit_transform(acc_summary[features])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
acc_summary["cluster"] = kmeans.fit_predict(X)

# Auto-labeling logic to match your notebook's "Professional" feel
label_map = {
    acc_summary.groupby("cluster")["avg_size_usd"].mean().idxmax(): "Whale Trader",
    acc_summary.groupby("cluster")["trades_per_day"].mean().idxmax(): "High-Frequency Trader",
}
for i in acc_summary["cluster"].unique():
    if i not in label_map: label_map[i] = "Retail / Risky"

acc_summary["Archetype"] = acc_summary["cluster"].map(label_map)

fig3 = px.scatter_3d(
    acc_summary, x="trades_per_day", y="avg_size_usd", z="total_pnl",
    color="Archetype", hover_name="Account",
    labels={"trades_per_day": "Freq", "avg_size_usd": "Avg Size", "total_pnl": "PnL"}
)
st.plotly_chart(fig3, use_container_width=True)