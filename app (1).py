import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Risk Decision Assistant", layout="wide", page_icon="📊")

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#1a1a2e; font-size:2.2rem; margin-bottom:0'>
        📊 Portfolio Risk Decision Assistant
    </h1>
    <p style='text-align:center; color:#666; margin-top:4px; font-size:1rem'>
        Quantitative Portfolio Risk Analytics for Retail Investors.
    </p>
    <hr style='margin:16px 0'>
""", unsafe_allow_html=True)

# ─── DATA ─────────────────────────────────────────────────────────────────────
stock_options = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "HCLTECH.NS",
    "WIPRO.NS", "BAJFINANCE.NS", "MARUTI.NS", "ONGC.NS", "NTPC.NS",
    "POWERGRID.NS", "SUNPHARMA.NS", "DRREDDY.NS", "ASIANPAINT.NS", "TITAN.NS"
]

sector_map = {
    "RELIANCE.NS": "Energy", "TCS.NS": "IT", "INFY.NS": "IT",
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking",
    "ITC.NS": "FMCG", "LT.NS": "Infrastructure", "AXISBANK.NS": "Banking",
    "HCLTECH.NS": "IT", "WIPRO.NS": "IT", "BAJFINANCE.NS": "Banking",
    "MARUTI.NS": "Auto", "ONGC.NS": "Energy", "NTPC.NS": "Energy",
    "POWERGRID.NS": "Infrastructure", "SUNPHARMA.NS": "Pharma",
    "DRREDDY.NS": "Pharma", "ASIANPAINT.NS": "FMCG", "TITAN.NS": "Consumer"
}

# ─── SIDEBAR ───────────────────────────────────────────────────────────────
st.sidebar.title("🏗️ Build Your Portfolio")

selected_stocks = st.sidebar.multiselect(
    "Select Stocks (min 2)",
    stock_options,
    default=["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS"]
)

if not selected_stocks or len(selected_stocks) < 2:
    st.warning("Please select at least 2 stocks to analyse portfolio risk.")
    st.stop()

portfolio = {}

for stock in selected_stocks:
    portfolio[stock] = st.sidebar.number_input(
        f"Quantity — {stock.replace('.NS', '')}",
        min_value=1,
        value=10,
        step=1,
        key=f"qty_{stock}"
    )

invested_amount = st.sidebar.number_input(
    "💰 Initial Investment Amount (₹) (₹)",
    min_value=1000,
    value=100000,
    step=1000
)

# ------------------------------------------------------------------
# WHAT IF SIMULATOR
# ------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 What If Simulator")

# Only show stocks NOT already in portfolio
available_stocks = sorted(
    [stock for stock in stock_options if stock not in selected_stocks]
)

# Safety check
if len(available_stocks) == 0:
    st.sidebar.info("All available stocks are already in your portfolio.")
    st.stop()

new_stock = st.sidebar.selectbox(
    "Test Adding This Stock",
    available_stocks
)

new_qty = st.sidebar.number_input(
    "Test Quantity",
    min_value=1,
    value=10,
    step=1
)
# ─── DATA LOAD ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_prices(tickers):
    data = yf.download(
        tickers, start="2024-01-01", end="2026-08-01",
        auto_adjust=True, progress=False
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]]
        close.columns = tickers
    return close.dropna(how="all")

tickers = list(portfolio.keys())
data = load_prices(tickers)

if data.empty or len(data) < 30:
    st.error("Could not load price data. Check your internet connection.")
    st.stop()

# ─── CALCULATIONS ─────────────────────────────────────────────────────────────
# ─── CALCULATIONS ─────────────────────────────────────────────────────────────

returns = data.pct_change().dropna()

# Keep only stocks that have valid downloaded price data
valid_tickers = [
    stock for stock in portfolio.keys()
    if stock in data.columns
]

if len(valid_tickers) < 2:
    st.error(
        "Not enough valid stock price data was returned. "
        "Please select different stocks and try again."
    )
    st.stop()

# Align price data and portfolio stocks
data = data[valid_tickers]
returns = returns[valid_tickers]

# Remove rows with missing values
returns = data.pct_change()

# Keep stocks that have enough valid historical observations
valid_tickers = [
    stock for stock in portfolio.keys()
    if stock in data.columns
    and data[stock].notna().sum() >= 30
]

if len(valid_tickers) < 2:
    st.error(
        "Not enough valid historical price data was returned. "
        "Please try again or select different stocks."
    )
    st.stop()

# Keep only valid portfolio stocks
data = data[valid_tickers]

# Calculate returns
returns = data.pct_change()

# Remove only rows where all selected stocks are missing
returns = returns.dropna(how="all")

# Fill small gaps using the previous available price
returns = returns.ffill().dropna()

if returns.empty:
    st.error("Unable to calculate returns from the downloaded market data.")
    st.stop()

# Latest prices
latest_prices = data.iloc[-1]

# Align quantities with the exact same stock order
quantities = pd.Series(portfolio).reindex(valid_tickers)

# Position value for each stock
position_values = latest_prices * quantities

# Total current portfolio value
portfolio_value = float(position_values.sum())

if portfolio_value <= 0:
    st.error("Portfolio value is zero. Check quantities.")
    st.stop()

# Portfolio weights
weights = position_values / portfolio_value

# Covariance matrix
cov_matrix = returns.cov()

# Force covariance matrix to use the same stocks and same order
cov_matrix = cov_matrix.reindex(
    index=valid_tickers,
    columns=valid_tickers
)

# Replace any remaining missing covariance values
cov_matrix = cov_matrix.fillna(0)

# Portfolio variance
w = weights.values.astype(float).reshape(-1, 1)
cov_values = cov_matrix.values.astype(float)

portfolio_variance = float(
    w.T @ cov_values @ w
)

# Portfolio volatility
portfolio_volatility = np.sqrt(
    max(portfolio_variance, 0)
)

# Portfolio return series
portfolio_returns_series = returns.dot(weights)
portfolio_mean = float(portfolio_returns_series.mean())

# 95% Parametric VaR
confidence_level = 1.65

daily_var = max(
    (
        confidence_level * portfolio_volatility
        - portfolio_mean
    ) * portfolio_value,
    0
)

# ─── RISK CONTRIBUTION ────────────────────────────────────────────────────────

marginal = cov_values @ weights.values

if portfolio_volatility > 0:
    risk_contrib = (
        weights.values
        * marginal
        / portfolio_volatility
    )
else:
    risk_contrib = np.zeros(len(weights))

risk_contribution = pd.Series(
    risk_contrib,
    index=weights.index
)

if risk_contribution.sum() != 0:
    risk_contribution_pct = (
        risk_contribution
        / risk_contribution.sum()
    ) * 100
else:
    risk_contribution_pct = pd.Series(
        0,
        index=weights.index
    )

# ─── INDIVIDUAL STOCK VAR ─────────────────────────────────────────────────────

individual_vars = []

for stock in valid_tickers:

    sv = returns[stock].std()

    stock_var = (
        confidence_level
        * sv
        * float(weights[stock])
        * portfolio_value
    )

    individual_vars.append(stock_var)

diversification_benefit = (
    sum(individual_vars) - daily_var
)

# ─── SECTOR WEIGHTS ───────────────────────────────────────────────────────────

sector_weights = {}

for stock, wt in weights.items():

    sector = sector_map.get(stock, "Other")

    sector_weights[sector] = (
        sector_weights.get(sector, 0)
        + float(wt) * 100
    )

sector_df = pd.DataFrame({
    "Sector": list(sector_weights.keys()),
    "Weight": [
        round(v, 2)
        for v in sector_weights.values()
    ]
})

if not sector_df.empty:

    top_sector = sector_df.loc[
        sector_df["Weight"].idxmax(),
        "Sector"
    ]

    top_sector_weight = sector_df["Weight"].max()

else:

    top_sector = "Other"
    top_sector_weight = 0

# ─── PROFIT / LOSS ────────────────────────────────────────────────────────────

current_pnl = (
    portfolio_value - invested_amount
)

pnl_pct = (
    current_pnl
    / invested_amount
) * 100

var_pct = (
    daily_var
    / portfolio_value
) * 100

# days_to_wipe = (
#     abs(current_pnl) / daily_var
#     if daily_var > 0 and current_pnl != 0
#     else 999
# )


# ─── HEALTH SCORE ─────────────────────────────────────────────────────────────
score = 100
if risk_contribution_pct.max() > 40:
    score -= 25
elif risk_contribution_pct.max() > 30:
    score -= 10
if top_sector_weight > 50:
    score -= 25
elif top_sector_weight > 40:
    score -= 10
if var_pct > 3:
    score -= 20
elif var_pct > 2:
    score -= 10
if len(selected_stocks) < 4:
    score -= 15
score = max(score, 0)



# ─── TOP METRICS ──────────────────────────────────────────────────────────────
st.markdown("### 📌 Portfolio Snapshot")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio Value", f"₹{portfolio_value:,.0f}")
c2.metric("Profit / Loss", f"₹{current_pnl:,.0f}", delta=f"{pnl_pct:+.1f}%")
c3.metric("Daily VaR (95%)", f"₹{daily_var:,.0f}", delta=f"{var_pct:.1f}% of portfolio", delta_color="inverse")
#c4.metric("Sharpe Ratio", f"{sharpe:.2f}")
c4.metric("Health Score", f"{score}/100")
if score >= 80:
    st.success("🟢 Well Diversified")
elif score >= 60:
    st.warning("🟡 Moderately Diversified")
else:
    st.error("🔴 Highly Concentrated")
#c5.metric("Health label", f"{health_label}")

st.markdown("---")



# ─── PER STOCK RISK BREAKDOWN ─────────────────────────────────────────────────
st.markdown("### 💰 Per Stock Risk Breakdown")
st.caption("See exactly how much risk each stock adds in rupees — not just percentages.")

stock_risk_table = pd.DataFrame({
    "Stock": [s.replace(".NS", "") for s in weights.index],
    "Amount Invested (₹)": (weights.values * portfolio_value).round(0).astype(int),
    "Portfolio Weight (%)": (weights.values * 100).round(1),
    "Risk Contribution (%)": risk_contribution_pct.values.round(1),
    "Daily Risk in ₹": (risk_contribution_pct.values / 100 * daily_var).round(0).astype(int),
})


st.dataframe(
    stock_risk_table.sort_values("Risk Contribution (%)", ascending=False),
    use_container_width=True
)
#st.caption("If Risk Contribution % is much higher than Portfolio Weight % — that stock is punching above its weight in risk.")

st.markdown("---")

# ─── RISK CONTRIBUTION CHART ──────────────────────────────────────────────────
st.markdown("### 📊 Risk Contribution vs Portfolio Weight")
risk_df = pd.DataFrame({
    "Stock": [s.replace(".NS", "") for s in risk_contribution_pct.index],
    "Risk Contribution (%)": risk_contribution_pct.values.round(2),
    "Portfolio Weight (%)": (weights.values * 100).round(2)
})
fig_risk = go.Figure()
fig_risk.add_trace(go.Bar(
    x=risk_df["Stock"], y=risk_df["Risk Contribution (%)"],
    name="Risk Contribution", marker_color="#e63946",
    text=risk_df["Risk Contribution (%)"].apply(lambda x: f"{x:.1f}%"),
    textposition="outside"
))
fig_risk.add_trace(go.Bar(
    x=risk_df["Stock"], y=risk_df["Portfolio Weight (%)"],
    name="Portfolio Weight", marker_color="#457b9d",
    text=risk_df["Portfolio Weight (%)"].apply(lambda x: f"{x:.1f}%"),
    textposition="outside"
))
fig_risk.update_layout(
    barmode="group", height=400,
    yaxis_title="Percentage (%)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    title="Red bar much higher than Blue = that stock is your biggest risk source"
)
st.plotly_chart(fig_risk, use_container_width=True)

st.markdown("---")

# ─── DIVERSIFICATION BENEFIT ──────────────────────────────────────────────────
st.markdown("### 🔀 Diversification Benefit")
col_d1, col_d2, col_d3 = st.columns(3)
col_d1.metric("Sum of Individual VaRs", f"₹{sum(individual_vars):,.0f}")
col_d2.metric("Actual Portfolio VaR", f"₹{daily_var:,.0f}")
col_d3.metric("₹ Saved by Diversification", f"₹{diversification_benefit:,.0f}")
if diversification_benefit > 0:
    st.success(
        f"Your diversification is saving you ₹{diversification_benefit:,.0f} in daily risk. "
        f"If you held each stock independently your total daily risk would be ₹{sum(individual_vars):,.0f}."
    )
else:
    st.warning("Your stocks are highly correlated. Diversification is providing minimal benefit.")

st.markdown("---")

# ─── SECTOR EXPOSURE ──────────────────────────────────────────────────────────
st.markdown("### 🏭 Sector and Stock Allocation")
col_s1, col_s2 = st.columns(2)
with col_s1:
    sec_fig = px.pie(sector_df, names="Sector", values="Weight", hole=0.45, title="Sector Allocation")
    sec_fig.update_traces(textinfo="label+percent")
    st.plotly_chart(sec_fig, use_container_width=True)
with col_s2:
    alloc_df = pd.DataFrame({
        "Stock": [s.replace(".NS", "") for s in weights.index],
        "Weight": (weights.values * 100).round(2)
    })
    alloc_fig = px.pie(alloc_df, names="Stock", values="Weight", hole=0.45, title="Stock Allocation")
    alloc_fig.update_traces(textinfo="label+percent")
    st.plotly_chart(alloc_fig, use_container_width=True)

if top_sector_weight > 50:
    st.error(f"High Sector Concentration: {top_sector} is {top_sector_weight:.1f}% of your portfolio. Safe limit is below 40-50%.")
elif top_sector_weight > 40:
    st.warning(f"Moderate Concentration: {top_sector} is {top_sector_weight:.1f}% of your portfolio. Consider diversifying.")

st.markdown("---")

# ─── CORRELATION HEATMAP ──────────────────────────────────────────────────────
st.markdown("### 🔥 Stock Correlation Heatmap")
st.caption("🔴 Values close to 1.0 = stocks move together = less diversification benefit")
st.caption("🟢 Values close to 0 or negative = stocks move independently = better diversification")
corr_matrix = returns.corr()
corr_matrix.columns = [c.replace(".NS", "") for c in corr_matrix.columns]
corr_matrix.index = [c.replace(".NS", "") for c in corr_matrix.index]
heatmap_fig = px.imshow(
    corr_matrix, text_auto=".2f",
    color_continuous_scale="RdBu_r", zmin=-1, zmax=1
)
heatmap_fig.update_layout(height=500)
st.plotly_chart(heatmap_fig, use_container_width=True)

st.markdown("---")

# ─── STRESS TESTING ───────────────────────────────────────────────────────────
st.markdown("### 💥 Stress Testing — What If Market Crashes?")

scenario_shocks = {
    "Minor Market Correction": {
        "IT": -0.03, "Banking": -0.04, "Energy": -0.02,
        "FMCG": -0.01, "Infrastructure": -0.03, "Auto": -0.03,
        "Pharma": -0.01, "Consumer": -0.02, "Other": -0.03
    },
    "Moderate Market Fall": {
        "IT": -0.05, "Banking": -0.06, "Energy": -0.04,
        "FMCG": -0.02, "Infrastructure": -0.05, "Auto": -0.05,
        "Pharma": -0.02, "Consumer": -0.04, "Other": -0.05
    },
    "Market -10% (Sharp Correction)": {
        "IT": -0.10, "Banking": -0.12, "Energy": -0.08,
        "FMCG": -0.05, "Infrastructure": -0.10, "Auto": -0.10,
        "Pharma": -0.04, "Consumer": -0.08, "Other": -0.10
    },
    "COVID Style Crash (Severe)": {
        "IT": -0.12, "Banking": -0.18, "Energy": -0.15,
        "FMCG": -0.08, "Infrastructure": -0.14, "Auto": -0.16,
        "Pharma": 0.05, "Consumer": -0.10, "Other": -0.14
    },
    "Interest Rate Hike (IT Heavy Fall)": {
        "IT": -0.14, "Banking": -0.05, "Energy": -0.03,
        "FMCG": -0.02, "Infrastructure": -0.04, "Auto": -0.06,
        "Pharma": -0.02, "Consumer": -0.05, "Other": -0.05
    },
    "Budget Shock — Feb 2020": {
        "IT": -0.04, "Banking": -0.08, "Energy": -0.05,
        "FMCG": -0.03, "Infrastructure": -0.07, "Auto": -0.06,
        "Pharma": -0.02, "Consumer": -0.04, "Other": -0.05
    }
}

scenario = st.selectbox("Select Market Scenario", list(scenario_shocks.keys()))
st.caption(
    "Sector-specific shocks are applied because different sectors react differently during market events."
)
shocks = scenario_shocks[scenario]

total_stressed_value = 0
stress_breakdown = []
for stock, wt in weights.items():
    sector = sector_map.get(stock, "Other")
    shock = shocks.get(sector, -0.05)
    stock_value = float(wt) * portfolio_value
    stressed_stock_value = stock_value * (1 + shock)
    stock_loss = stock_value - stressed_stock_value
    total_stressed_value += stressed_stock_value
    stress_breakdown.append({
        "Stock": stock.replace(".NS", ""),
        "Current Value (₹)": round(stock_value),
        "Stressed Value (₹)": round(stressed_stock_value),
        "Loss (₹)": round(stock_loss),
        "Shock Applied": f"{shock*100:.1f}%"
    })

total_loss = portfolio_value - total_stressed_value
loss_pct = (total_loss / portfolio_value) * 100

st.error(f"In this scenario your portfolio loses approximately ₹{total_loss:,.0f} ({loss_pct:.1f}% of value)")

if current_pnl > 0:
    remaining_profit = current_pnl - total_loss
    if remaining_profit > 0:
        st.warning(f"Your current profit of ₹{current_pnl:,.0f} reduces to ₹{remaining_profit:,.0f} after this crash.")
    else:
        st.error(f"This crash would wipe your profit and put you in loss of ₹{abs(remaining_profit):,.0f}.")

stress_df = pd.DataFrame(stress_breakdown).sort_values("Loss (₹)", ascending=False)
st.dataframe(stress_df, use_container_width=True)

st.markdown("---")
# ─── DECISION SUMMARY CARD ────────────────────────────────────────────────────
st.markdown("### 🎯 Portfolio Risk Insights — What Should You Do?")
st.caption("Based on your portfolio risk, profit/loss status, and concentration analysis.")

decisions = []

top_risk_stock = risk_contribution_pct.idxmax()
top_risk_val = risk_contribution_pct.max()
top_stock_weight_pct = float(weights[top_risk_stock]) * 100

if top_risk_val > 40:
    decisions.append(("REDUCE", top_risk_stock.replace(".NS", ""),
        f"Drives {top_risk_val:.1f}% of portfolio risk but is only {top_stock_weight_pct:.1f}% of your money. "
        f"Reducing this position will lower your overall risk significantly."))
else:
    decisions.append(("HOLD", top_risk_stock.replace(".NS", ""),
        f"Risk contribution of {top_risk_val:.1f}% is within acceptable range. No immediate action needed."))

if top_sector_weight > 50:
    decisions.append(("AVOID ADDING", f"{top_sector} Sector",
        f"You are {top_sector_weight:.1f}% concentrated in {top_sector}. "
        f"Avoid adding more {top_sector} stocks."))

if var_pct > 3:
    decisions.append(("HIGH RISK", "Overall Portfolio",
        f"Daily VaR of {var_pct:.1f}% is above safe limit of 2-3%. "
        f"On a bad day you could lose ₹{daily_var:,.0f}. Consider rebalancing."))
else:
    decisions.append(("ACCEPTABLE", "Overall Portfolio",
        f"Portfolio risk level is within safe range at {var_pct:.1f}% daily VaR."))

if current_pnl > 0:

    if top_risk_val > 35:

        decisions.append((
            "BOOK PARTIAL PROFIT",
            "Profit Alert",
            f"You are currently in profit of ₹{current_pnl:,.0f}. "
            f"{top_risk_stock.replace('.NS','')} contributes {top_risk_val:.1f}% of portfolio risk. "
            f"If you plan to reduce exposure, this stock is the first candidate."
        ))

    else:

        decisions.append((
            "HOLD",
            "Profit Status",
            f"You are currently in profit of ₹{current_pnl:,.0f}. "
            f"Portfolio risk is within acceptable limits."
        ))

elif current_pnl < 0:

    decisions.append((
        "AVOID AVERAGING DOWN",
        "Loss Alert",
        f"You are currently in a loss of ₹{abs(current_pnl):,.0f}. "
        f"Avoid increasing exposure until portfolio risk becomes more balanced."
    ))

color_map = {
    "HOLD": "green", "REDUCE": "orange", "AVOID ADDING": "red",
    "HIGH RISK": "red", "ACCEPTABLE": "green",
    "BOOK PARTIAL PROFIT": "orange", "AVOID AVERAGING DOWN": "red"
}

for decision, subject, reason in decisions:
    color = color_map.get(decision, "gray")
    if color == "green":
        st.success(f"**{subject}** → **{decision}** | {reason}")
    elif color == "orange":
        st.warning(f"**{subject}** → **{decision}** | {reason}")
    else:
        st.error(f"**{subject}** → **{decision}** | {reason}")

st.markdown("---")

# ─── WHAT IF SIMULATOR ────────────────────────────────────────────────────────
st.markdown("### 🧪 What If Simulator — Should You Add This Stock?")
st.caption(f"Testing: Add {new_qty} shares of {new_stock.replace('.NS', '')} to your portfolio")

simulated_portfolio = portfolio.copy()
simulated_portfolio[new_stock] = simulated_portfolio.get(new_stock, 0) + new_qty
sim_tickers = list(simulated_portfolio.keys())
sim_data = load_prices(sim_tickers)

if not sim_data.empty and len(sim_data) > 30:

    sim_returns = sim_data.pct_change().dropna()
    sim_latest = sim_data.iloc[-1]
    sim_qty = pd.Series(simulated_portfolio)
    sim_pos = sim_latest * sim_qty
    sim_val = float(sim_pos.sum())
    sim_wts = sim_pos / sim_val
    sim_cov = sim_returns.cov()
    sw = sim_wts.values.reshape(-1, 1)
    sim_var_port = float(sw.T @ sim_cov.values @ sw)
    sim_vol = np.sqrt(max(sim_var_port, 0))
    sim_mean = float(sim_returns.dot(sim_wts).mean())
    new_var = max((confidence_level * sim_vol - sim_mean) * sim_val, 0)
    var_diff = new_var - daily_var

    new_sector = sector_map.get(new_stock, "Other")

    sector_before = sector_weights.get(new_sector, 0)

    sector_after = (
        sector_before +
        (float(sim_latest.get(new_stock, 0)) * new_qty / sim_val * 100)
    )

    wc1, wc2, wc3 = st.columns(3)

    wc1.metric("Current VaR", f"₹{daily_var:,.0f}")

    wc2.metric(
        "New VaR",
        f"₹{new_var:,.0f}",
        delta=f"₹{var_diff:+,.0f}",
        delta_color="inverse"
    )

    wc3.metric(
        f"{new_sector} Exposure",
        f"{sector_after:.1f}%",
        delta=f"{sector_before:.1f}% → {sector_after:.1f}%"
    )

else:
    st.warning("Could not load simulation data.")

# ─── RISK ALERTS ──────────────────────────────────────────────────────────────
st.markdown("### 🚨 Risk Alerts")
alert_count = 0
if risk_contribution_pct.max() > 40:
    st.warning(f"{top_risk_stock.replace('.NS','')} contributes {risk_contribution_pct.max():.1f}% of portfolio risk")
    alert_count += 1
if var_pct > 3:
    st.warning(f"Portfolio VaR ({var_pct:.1f}%) exceeds safe 3% threshold")
    alert_count += 1
if top_sector_weight > 50:
    st.error(f"{top_sector} sector concentration is {top_sector_weight:.1f}% — above safe limit")
    alert_count += 1
if len(selected_stocks) < 5:
    st.warning(f"Only {len(selected_stocks)} stocks — consider adding more for better diversification")
    alert_count += 1
if alert_count == 0:
    st.success("No major risk alerts. Portfolio looks balanced.")

st.markdown("---")
st.caption("Portfolio Risk Decision Assistant | Developed using Python, Streamlit and Quantitative Risk Analytics. | Market data from Yahoo Finance.| For educational purposes only.")
