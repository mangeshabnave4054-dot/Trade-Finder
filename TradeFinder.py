import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures

# --- APP SETUP ---
st.set_page_config(page_title="Nifty 500 Master Scanner", layout="wide")

# --- SHARED FUNCTIONS ---
@st.cache_data(ttl=86400)
def get_nifty500_symbols():
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        df = pd.read_csv(url)
        return [f"{s}.NS" for s in df['Symbol'].tolist()]
    except Exception:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
                "TATAMOTORS.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS", "ITC.NS"]

def resample_to_weekly(df_daily):
    """DATA MINIMIZATION: Instantly converts Daily candles to Weekly candles mathematically."""
    df_weekly = df_daily.resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    return df_weekly

def calculate_custom_rs(stock_series, benchmark_series, period=21):
    stock_ret = stock_series / stock_series.shift(period)
    bench_ret = benchmark_series / benchmark_series.shift(period)
    return (stock_ret / bench_ret) - 1.0

# --- HELPER: GAPLESS 0-CROSSING SPLIT ---
def split_series_at_zero(series):
    x_pos, y_pos = [], []
    x_neg, y_neg = [], []
    dates, vals = series.index, series.values
    for i in range(len(vals)):
        curr_x, curr_y = dates[i], vals[i]
        if i == 0:
            (x_pos if curr_y >= 0 else x_neg).append(curr_x)
            (y_pos if curr_y >= 0 else y_neg).append(curr_y)
            continue
        prev_x, prev_y = dates[i-1], vals[i-1]
        if (prev_y < 0 and curr_y > 0) or (prev_y > 0 and curr_y < 0):
            cross_time = prev_x + (curr_x - prev_x) * (abs(prev_y) / (abs(prev_y) + abs(curr_y)))
            x_pos.extend([cross_time, curr_x if curr_y >= 0 else None])
            y_pos.extend([0, curr_y if curr_y >= 0 else None])
            x_neg.extend([cross_time, curr_x if curr_y < 0 else None])
            y_neg.extend([0, curr_y if curr_y < 0 else None])
        else:
            if curr_y >= 0:
                x_pos.append(curr_x); y_pos.append(curr_y)
                x_neg.append(curr_x); y_neg.append(None)
            else:
                x_neg.append(curr_x); y_neg.append(curr_y)
                x_pos.append(curr_x); y_pos.append(None)
    return x_pos, y_pos, x_neg, y_neg

def calculate_ivr_vixfix(df, vix_len=22, rank_len=252):
    highest_close = df['Close'].rolling(window=vix_len, min_periods=1).max()
    vix_fix = np.where(highest_close != 0, ((highest_close - df['Low']) / highest_close) * 100.0, 0.0)
    vix_fix_series = pd.Series(vix_fix, index=df.index)
    lowest_vix = vix_fix_series.rolling(window=rank_len, min_periods=1).min()
    highest_vix = vix_fix_series.rolling(window=rank_len, min_periods=1).max()
    vix_range = highest_vix - lowest_vix
    ivr = np.where(vix_range != 0, ((vix_fix_series - lowest_vix) / vix_range) * 100.0, 0.0)
    return pd.Series(ivr, index=df.index, name="IVR_VixFix").fillna(0)

def calculate_tv_rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1.0 / length, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / length, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.where(roll_down == 0, 100.0, np.where(roll_up == 0, 0.0, rsi))
    return pd.Series(rsi, index=series.index, name="RSI14")


# --- WORKER FUNCTIONS FOR MULTITHREADING ---
def scan_strategy_1_worker(sym, nifty_weekly, rs_length, rsi_min):
    """Processes a single stock for Strategy 1"""
    try:
        # Download Daily Data Once (Data Minimization)
        df_d = yf.download(sym, period="3y", interval="1d", progress=False)
        if df_d.empty or len(df_d) < 60: return None
        if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)

        # Convert to Weekly
        df_w = resample_to_weekly(df_d)
        if len(df_w) < 30: return None

        df_w['EMA21'] = ta.ema(df_w['Close'], length=21)
        df_w['EMA55'] = ta.ema(df_w['Close'], length=55)
        df_w['RSI14'] = ta.rsi(df_w['Close'], length=14)

        ema21_w0 = float(df_w['EMA21'].iloc[-1])
        ema55_w0 = float(df_w['EMA55'].iloc[-1])
        ema21_w1 = float(df_w['EMA21'].iloc[-2])
        ema55_w1 = float(df_w['EMA55'].iloc[-2])
        ema21_w2 = float(df_w['EMA21'].iloc[-3])
        ema55_w2 = float(df_w['EMA55'].iloc[-3])
        w_rsi = float(df_w['RSI14'].iloc[-1])

        cross_this_week = (ema21_w1 <= ema55_w1) and (ema21_w0 > ema55_w0)
        cross_last_week = (ema21_w2 <= ema55_w2) and (ema21_w1 > ema55_w1)
        bullish_crossover = cross_this_week or cross_last_week
        rsi_condition = w_rsi >= rsi_min

        if not (bullish_crossover and rsi_condition): return None

        common_idx = df_w.index.intersection(nifty_weekly.index)
        rs = calculate_custom_rs(df_w.loc[common_idx, 'Close'], nifty_weekly.loc[common_idx, 'Close'], rs_length)
        if rs.empty: return None
        w_rs = float(rs.iloc[-1])
        curr_price = float(df_d['Close'].iloc[-1])

        recency_score = 1 if cross_this_week else 2
        crossover_tag = "🔥 Current Week" if cross_this_week else "📅 Previous Week"

        return {
            "Symbol": sym.replace(".NS", ""),
            "Close Price": round(curr_price, 2),
            "Signal Recency": crossover_tag,
            "Weekly RSI": round(w_rsi, 1),
            "Weekly RS %": round(w_rs * 100, 2),
            "_recency": recency_score
        }
    except Exception:
        return None

def scan_strategy_2_worker(sym, ivr_threshold, rsi_threshold):
    """Processes a single stock for Strategy 2"""
    try:
        df_d = yf.download(sym, period="2y", interval="1d", progress=False)
        if df_d.empty or len(df_d) < 30: return None
        if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)

        df_d['IVR_VixFix'] = calculate_ivr_vixfix(df_d, vix_len=22, rank_len=252)
        df_d['RSI14'] = calculate_tv_rsi(df_d['Close'], length=14)
        
        latest_ivr = float(df_d['IVR_VixFix'].iloc[-1])
        latest_rsi = float(df_d['RSI14'].iloc[-1])
        curr_price = float(df_d['Close'].iloc[-1])
        
        if latest_ivr > ivr_threshold and latest_rsi < rsi_threshold:
            ivr_10d_high = float(df_d['IVR_VixFix'].rolling(window=10, min_periods=1).max().iloc[-1])
            ivr_10d_low  = float(df_d['IVR_VixFix'].rolling(window=10, min_periods=1).min().iloc[-1])
            return {
                "Symbol": sym.replace(".NS", ""),
                "Close Price": round(curr_price, 2),
                "Daily RSI": round(latest_rsi, 2),
                "IV Rank (VixFix) %": round(latest_ivr, 2),
                "VixFix (10D Low)": round(ivr_10d_low, 2),
                "VixFix (10D High)": round(ivr_10d_high, 2)
            }
        return None
    except Exception:
        return None


# --- STRATEGY SELECTOR ---
st.sidebar.header("Navigation")
strategy_choice = st.sidebar.radio(
    "Select Strategy:", 
    ["Strategy 1: EMA Crossover & RS Trend", "Strategy 2: High IV & Oversold"]
)
st.sidebar.markdown("---")


# =====================================================================
# STRATEGY 1: EMA Crossover & RS TREND SETUP
# =====================================================================
if strategy_choice == "Strategy 1: EMA Crossover & RS Trend":
    st.title("🎯 Nifty 500 Swing Trading Screener")
    st.markdown("Strategy: **Weekly 21/55 EMA**, ** RS (21 vs Nifty 50)**, and **Daily Breakout Setup**.")

    st.sidebar.header("Strategy 1 Settings")
    rs_length = st.sidebar.number_input("RS Lookback Period", value=21)
    rsi_min = st.sidebar.slider("Minimum Weekly RSI", 40, 70, 50)
    run_scan_1 = st.sidebar.button("🚀 Run Strategy 1 Scan")

    if run_scan_1:
        # NIFTY Benchmark Checks (Data Minimization applied here too)
        with st.spinner("Checking NIFTY 50 Regime..."):
            nifty_daily = yf.download("^NSEI", period="3y", interval="1d", progress=False)
            if isinstance(nifty_daily.columns, pd.MultiIndex): nifty_daily.columns = nifty_daily.columns.get_level_values(0)
            
            nifty_daily['EMA200'] = ta.ema(nifty_daily['Close'], length=200)
            latest_nifty_close = float(nifty_daily['Close'].iloc[-1])
            latest_nifty_ema200 = float(nifty_daily['EMA200'].iloc[-1])
            
            # Prepare precise Weekly benchmark for RS calculation
            nifty_weekly = resample_to_weekly(nifty_daily)

        if latest_nifty_close < latest_nifty_ema200:
            st.error(f"⚠️ Market Filter Active: NIFTY 50 ({latest_nifty_close:.1f}) is BELOW 200 EMA ({latest_nifty_ema200:.1f}). Long trades paused.")
            st.stop()
        else:
            st.success(f"✅ Market Regime Bullish: NIFTY 50 ({latest_nifty_close:.1f}) is ABOVE 200 EMA ({latest_nifty_ema200:.1f}). Scanning stocks...")
        
        symbols = get_nifty500_symbols()
        progress_bar = st.progress(0)
        results = []

        # --- MULTITHREADING IMPLEMENTATION ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(scan_strategy_1_worker, sym, nifty_weekly, rs_length, rsi_min): sym for sym in symbols}
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                # OPTIMIZATION: Only update UI every 15 stocks to prevent frontend lag
                if i % 15 == 0 or i == len(symbols) - 1:
                    progress_bar.progress((i + 1) / len(symbols))
                
                res = future.result()
                if res:
                    results.append(res)

        progress_bar.empty()

        if results:
            res_df = pd.DataFrame(results).sort_values(by=["_recency", "Weekly RS %"], ascending=[True, False])
            res_df = res_df.drop(columns=["_recency"])
            st.session_state['scan_results_1'] = res_df
            st.session_state['nifty_weekly_1'] = nifty_weekly
        else:
            st.warning("No stocks matched Strategy 1 conditions today.")

    # Display Watchlist
    if 'scan_results_1' in st.session_state:
        res_df = st.session_state['scan_results_1']
        st.subheader(f"Strategy 1 Watchlist ({len(res_df)} Stocks)")
        st.dataframe(res_df, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Stock Chart & RS Inspector")
        selected_stock = st.selectbox("Select stock to inspect (Strat 1):", res_df['Symbol'].tolist(), key="strat1_select")
        
        if selected_stock:
            chart_tf = st.radio("Select Chart Timeframe:", ["Weekly (Strategy View)", "Daily (Tactical View)"], horizontal=True, key="strat1_tf")
            
            # --- DATA MINIMIZATION ON CHARTING ---
            df_chart = yf.download(f"{selected_stock}.NS", period="3y", interval="1d", progress=False)
            if isinstance(df_chart.columns, pd.MultiIndex): df_chart.columns = df_chart.columns.get_level_values(0)
            
            bench_df = yf.download("^NSEI", period="3y", interval="1d", progress=False)
            if isinstance(bench_df.columns, pd.MultiIndex): bench_df.columns = bench_df.columns.get_level_values(0)

            # Resample dynamically without re-downloading
            if "Weekly" in chart_tf:
                df_chart = resample_to_weekly(df_chart)
                bench_df = resample_to_weekly(bench_df)
                
            df_chart['EMA21'] = ta.ema(df_chart['Close'], length=21)
            df_chart['EMA55'] = ta.ema(df_chart['Close'], length=55)
            df_chart['RSI14'] = ta.rsi(df_chart['Close'], length=14)

            common_idx = df_chart.index.intersection(bench_df.index)
            rs = calculate_custom_rs(df_chart.loc[common_idx, 'Close'], bench_df.loc[common_idx, 'Close'], rs_length)
            x_pos, y_pos, x_neg, y_neg = split_series_at_zero(rs.dropna())

            # Formatting
            total_bars = len(df_chart)
            view_window = 180
            start_date = df_chart.index[-view_window] if total_bars > view_window else df_chart.index[0]
            end_date_padded = df_chart.index[-1] + pd.Timedelta(days=10)

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25],
                subplot_titles=(f"{selected_stock} Price & EMAs", "Relative Strength (vs Nifty 50)", "Relative Strength Index (14)")
            )
            
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA21'], line=dict(color='blue', width=1.5), name='21 EMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA55'], line=dict(color='orange', width=1.5), name='55 EMA'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=x_pos, y=y_pos, mode='lines', line=dict(color='green', width=2), connectgaps=False, name='RS (+ve)'), row=2, col=1)
            fig.add_trace(go.Scatter(x=x_neg, y=y_neg, mode='lines', line=dict(color='red', width=2), connectgaps=False, name='RS (-ve)'), row=2, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI14'], line=dict(color='#7E57C2', width=2), name='RSI 14'), row=3, col=1)
            fig.add_trace(go.Scatter(x=[df_chart.index[0], end_date_padded], y=[70, 70], mode="lines", line=dict(color="#0037FF", dash="dash", width=1), hoverinfo="skip"), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=[df_chart.index[0], end_date_padded], y=[30, 30], mode="lines", line=dict(color="#0037FF", dash="dash", width=1), 
                fill="tonexty", fillcolor="rgba(126, 87, 194, 0.12)", hoverinfo="skip"
            ), row=3, col=1)
            fig.add_trace(go.Scatter(x=[df_chart.index[0], end_date_padded], y=[50, 50], mode="lines", line=dict(color="rgba(120, 123, 134, 0.5)", dash="dot", width=1), hoverinfo="skip"), row=3, col=1)

            # Conditionally apply rangebreaks based on timeframe
            if "Weekly" in chart_tf:
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.15)', range=[start_date, end_date_padded])
            else:
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.15)', range=[start_date, end_date_padded])
            fig.update_layout(height=850, xaxis_rangeslider_visible=False, showlegend=False, bargap=0.15, margin=dict(t=40, b=20, l=20, r=20), font=dict(color='#d1d4dc'))
            fig.update_yaxes(range=[0, 100], row=3, col=1)

            st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# STRATEGY 2: VOLATILITY & OVERSOLD SETUP
# =====================================================================
elif strategy_choice == "Strategy 2: High IV & Oversold":
    st.title("⚡ Nifty 500 Two-Stage Volatility Scanner")
    st.markdown("Strategy: **Stage 1:** IV Rank > Threshold (High Volatility) ➡️ **Stage 2:** RSI < Threshold (Oversold).")

    st.sidebar.header("Strategy 2 Settings")
    ivr_threshold = st.sidebar.number_input("Stage 1: Min IV Rank", min_value=0, max_value=100, value=50, step=5)
    rsi_threshold = st.sidebar.number_input("Stage 2: Max Daily RSI", min_value=0, max_value=100, value=31, step=5)
    run_scan_2 = st.sidebar.button("🚀 Run Strategy 2 Scan")

    if run_scan_2:
        symbols = get_nifty500_symbols()
        progress_bar = st.progress(0)
        results = []

        # --- MULTITHREADING IMPLEMENTATION ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(scan_strategy_2_worker, sym, ivr_threshold, rsi_threshold): sym for sym in symbols}
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                progress_bar.progress((i + 1) / len(symbols))
                res = future.result()
                if res:
                    results.append(res)

        progress_bar.empty()

        if results:
            res_df = pd.DataFrame(results).sort_values(by="IV Rank (VixFix) %", ascending=False)
            st.session_state['scan_results_2'] = res_df
        else:
            st.warning(f"No stocks found meeting both criteria (IVR > {ivr_threshold} & RSI < {rsi_threshold}) today.")

    if 'scan_results_2' in st.session_state:
        res_df = st.session_state['scan_results_2']
        st.subheader(f"Strategy 2 Watchlist ({len(res_df)} Stocks)")
        st.dataframe(res_df, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Volatility & RSI Inspector")
        
        selected_stock = st.selectbox("Select stock to inspect (Strat 2):", res_df['Symbol'].tolist(), key="strat2_select")
        
        if selected_stock:
            df_chart = yf.download(f"{selected_stock}.NS", period="2y", interval="1d", progress=False)
            if isinstance(df_chart.columns, pd.MultiIndex): df_chart.columns = df_chart.columns.get_level_values(0)
                
            df_chart['IVR_VixFix'] = calculate_ivr_vixfix(df_chart, vix_len=22, rank_len=252)
            df_chart['RSI14'] = calculate_tv_rsi(df_chart['Close'], length=14)

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.54, 0.23, 0.23],
                subplot_titles=(f"{selected_stock} Daily Price", "SegaRKO IV Rank [VixFix-Based]", "Relative Strength Index (14)")
            )
            
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Price'), row=1, col=1)
            
            total_bars = len(df_chart)
            view_window = 180
            start_date = df_chart.index[-view_window] if total_bars > view_window else df_chart.index[0]
            end_date_padded = df_chart.index[-1] + pd.Timedelta(days=10)

            bar_colors = ['#4CAF50' if val > 50 else '#F44336' for val in df_chart['IVR_VixFix']]
            fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['IVR_VixFix'], marker_color=bar_colors, name='IV Rank (VixFix)'), row=2, col=1)

            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI14'], line=dict(color='#7E57C2', width=2), name='RSI 14'), row=3, col=1)
            fig.add_trace(go.Scatter(x=[df_chart.index[0], end_date_padded], y=[70, 70], mode="lines", line=dict(color="#0037FF", dash="dash", width=1), hoverinfo="skip"), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=[df_chart.index[0], end_date_padded], y=[30, 30], mode="lines", line=dict(color="#0037FF", dash="dash", width=1), 
                fill="tonexty", fillcolor="rgba(126, 87, 194, 0.12)", hoverinfo="skip"
            ), row=3, col=1)
            fig.add_trace(go.Scatter(x=[df_chart.index[0], end_date_padded], y=[50, 50], mode="lines", line=dict(color="rgba(120, 123, 134, 0.5)", dash="dot", width=1), hoverinfo="skip"), row=3, col=1)

            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.15)', range=[start_date, end_date_padded])
            fig.update_layout(height=850, xaxis_rangeslider_visible=False, showlegend=False, bargap=0.15, margin=dict(t=40, b=20, l=20, r=20), font=dict(color='#d1d4dc'))
            fig.update_yaxes(range=[0, 100], row=2, col=1)
            fig.update_yaxes(range=[0, 100], row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)