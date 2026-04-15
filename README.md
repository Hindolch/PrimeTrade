# Trader Performance vs. Market Sentiment Analysis
### Data Science Intern Assignment — Round 0 | Primetrade.ai

This project analyzes the relationship between Bitcoin market sentiment (Fear/Greed Index) and trader behavior/performance on the Hyperliquid exchange. The goal is to identify statistically significant patterns and behavioral archetypes to inform smarter trading strategies.

## 🚀 Key Features
- **Statistical Rigor:** Validated performance shifts across sentiment regimes using Mann-Whitney U testing ($p < 0.0001$).
- **Behavioral Clustering:** Segmented accounts into professional archetypes (Whale Traders, High-Frequency Specialists, etc.) using K-Means.
- **Predictive Modeling:** Built a Gradient Boosting classifier to forecast next-day profitability based on sentiment and behavior.
- **Interactive Dashboard:** A full-stack Streamlit application for real-time data exploration and stress-testing of strategy rules.

## 📂 Project Structure
```text
.
├── data/                       # Contains the CSV datasets
├── primetrade_analysis.ipynb   # Core Jupyter Notebook (Research & Stats)
├── app.py                      # Streamlit Dashboard code
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🛠️ Installation & Setup

1. **Clone the repository or unzip the files:**
   ```bash
   cd primetrade-analytics
   ```

2. **Install Dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Analysis:**
   Open `primetrade_analysis.ipynb` in Jupyter Lab or VS Code to view the full research pipeline, statistical tests, and strategy derivations.

4. **Launch the Dashboard:**
   To explore the data interactively, run the following command:
   ```bash
   streamlit run app.py
   ```

## 📈 Summary of Insights
- **Fear Alpha:** Aggregate PnL is statistically 7x higher during Fear days compared to Greed days, suggesting that "Fear" provides superior liquidity for experienced traders.
- **Scaling Behavior:** Traders on Hyperliquid do not shy away from panic; they increase trade frequency by ~50% and average position sizes by ~63% during Fear regimes.
- **Margin Discipline:** Isolated-margin traders exhibited a 38% higher median PnL than cross-margin traders, highlighting the importance of risk-defined structures during volatility.

## 💡 Strategy Recommendations
- **Dynamic Sizing:** Increase capital allocation multiplier for "Sniper" archetypes during Extreme Fear.
- **Risk Guardrails:** Enforce strict leverage caps on "High-Risk" clusters during Greed periods to prevent drawdown from trend exhaustion.

