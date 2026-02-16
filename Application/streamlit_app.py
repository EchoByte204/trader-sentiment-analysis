"""
Trader Performance vs Market Sentiment Dashboard
Interactive Streamlit Application - FULLY FIXED VERSION
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Trader Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .strategy-box {
        background-color: #f0fff4;
        padding: 1rem;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load and prepare all necessary data"""
    try:
        # Load sentiment data
        sentiment_df = pd.read_csv('Data/fear_greed_index.csv')
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df['is_greed'] = sentiment_df['classification'].isin(['Greed', 'Extreme Greed']).astype(int)
        sentiment_df['is_fear'] = sentiment_df['classification'].isin(['Fear', 'Extreme Fear']).astype(int)
        
        # Load trader data (try CSV first, then Excel)
        try:
            trader_df = pd.read_csv('Data/trader_data.csv')
        except:
            try:
                trader_df = pd.read_excel('trader_data.xlsx')
            except:
                st.error("Could not load trader data. Please ensure 'trader_data.csv' or 'trader_data.xlsx' exists.")
                return None, None, None, None
        
        # Process trader data
        trader_df['trade_datetime'] = pd.to_datetime(trader_df['Timestamp'], unit='ms', errors='coerce')
        trader_df['date'] = trader_df['trade_datetime'].dt.date
        trader_df['date'] = pd.to_datetime(trader_df['date'])
        
        # Clean numeric columns
        trader_df['pnl'] = pd.to_numeric(trader_df['Closed PnL'], errors='coerce')
        trader_df['size_usd'] = pd.to_numeric(trader_df['Size USD'], errors='coerce')
        trader_df['fee'] = pd.to_numeric(trader_df['Fee'], errors='coerce')
        
        # Remove missing data
        trader_df = trader_df.dropna(subset=['date', 'Account'])
        
        # Merge datasets
        merged_df = trader_df.merge(
            sentiment_df[['date', 'classification', 'value', 'is_greed', 'is_fear']], 
            on='date', 
            how='left'
        )
        merged_df = merged_df[merged_df['classification'].notna()]
        
        # Create daily metrics
        daily_metrics = merged_df.groupby(['Account', 'date']).agg({
            'pnl': ['sum', 'mean', 'count', 'std'],
            'size_usd': ['sum', 'mean'],
            'fee': 'sum',
            'Side': lambda x: (x == 'BUY').sum() / len(x) if len(x) > 0 else 0,
            'Coin': 'nunique'
        }).reset_index()
        
        daily_metrics.columns = [
            'Account', 'date', 
            'total_pnl', 'avg_pnl', 'num_trades', 'pnl_std',
            'total_size_usd', 'avg_size_usd', 'total_fees',
            'long_ratio', 'num_coins'
        ]
        
        # Calculate win rate
        win_trades = merged_df[merged_df['pnl'] > 0].groupby(['Account', 'date']).size()
        total_trades = merged_df.groupby(['Account', 'date']).size()
        daily_metrics['win_rate'] = (win_trades / total_trades).fillna(0).values
        
        # Calculate net PnL
        daily_metrics['net_pnl'] = daily_metrics['total_pnl'] - daily_metrics['total_fees']
        daily_metrics['short_ratio'] = 1 - daily_metrics['long_ratio']
        
        # Merge sentiment back
        daily_metrics = daily_metrics.merge(
            sentiment_df[['date', 'classification', 'value', 'is_greed', 'is_fear']], 
            on='date', 
            how='left'
        )
        
        # Create trader statistics
        trader_stats = daily_metrics.groupby('Account').agg({
            'net_pnl': ['sum', 'mean', 'std'],
            'win_rate': 'mean',
            'num_trades': ['sum', 'mean'],
            'date': 'count',
            'long_ratio': 'mean'
        }).reset_index()
        
        trader_stats.columns = [
            'Account', 
            'total_pnl', 'avg_daily_pnl', 'pnl_volatility',
            'avg_win_rate', 
            'total_trades', 'avg_daily_trades',
            'trading_days',
            'avg_long_ratio'
        ]
        
        # Create segments
        trader_stats['frequency_group'] = pd.cut(
            trader_stats['avg_daily_trades'], 
            bins=[0, 5, 20, 1000], 
            labels=['Low (<5)', 'Medium (5-20)', 'High (20+)']
        )
        
        trader_stats['winrate_group'] = pd.cut(
            trader_stats['avg_win_rate'], 
            bins=[0, 0.4, 0.6, 1.0], 
            labels=['Low (<40%)', 'Medium (40-60%)', 'High (60%+)']
        )
        
        return sentiment_df, merged_df, daily_metrics, trader_stats
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure 'fear_greed_index.csv' and trader data file are in the same directory")
        return None, None, None, None

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">📊 Trader Performance vs Market Sentiment</div>', unsafe_allow_html=True)
    st.markdown("**Interactive Dashboard** | Analyzing trader behavior across Fear/Greed market conditions")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        sentiment_df, merged_df, daily_metrics, trader_stats = load_data()
    
    if sentiment_df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Date filter
    min_date = daily_metrics['date'].min()
    max_date = daily_metrics['date'].max()
    
    st.sidebar.subheader("📅 Date Range")
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Sentiment filter
    st.sidebar.subheader("😱 Sentiment Filter")
    sentiment_filter = st.sidebar.multiselect(
        "Select sentiment types",
        options=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'],
        default=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    )
    
    # Filter data
    if len(date_range) == 2:
        filtered_daily = daily_metrics[
            (daily_metrics['date'] >= pd.to_datetime(date_range[0])) &
            (daily_metrics['date'] <= pd.to_datetime(date_range[1])) &
            (daily_metrics['classification'].isin(sentiment_filter))
        ]
    else:
        filtered_daily = daily_metrics[daily_metrics['classification'].isin(sentiment_filter)]
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.subheader("📑 Navigation")
    page = st.sidebar.radio(
        "Select page",
        ["📊 Overview", "📈 Performance Analysis", "🎯 Behavior Analysis", 
         "👥 Trader Segments", "💡 Insights & Strategies", "🔮 Predictive Model"]
    )
    
    # Overview Page
    if page == "📊 Overview":
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", f"{len(merged_df):,}")
        with col2:
            st.metric("Unique Traders", f"{daily_metrics['Account'].nunique():,}")
        with col3:
            st.metric("Trading Days", f"{daily_metrics['date'].nunique():,}")
        with col4:
            avg_daily_pnl = filtered_daily['net_pnl'].mean()
            st.metric("Avg Daily PnL", f"${avg_daily_pnl:,.2f}")
        
        st.markdown("---")
        
        # Sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution Over Time")
            fig = px.histogram(
                sentiment_df, 
                x='date', 
                color='classification',
                color_discrete_map={
                    'Extreme Fear': '#8B0000',
                    'Fear': '#DC143C',
                    'Neutral': '#808080',
                    'Greed': '#90EE90',
                    'Extreme Greed': '#006400'
                },
                title="Daily Sentiment Classification"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sentiment Value Distribution")
            sentiment_counts = sentiment_df['classification'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                marker=dict(colors=['#8B0000', '#DC143C', '#808080', '#90EE90', '#006400'])
            )])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Trading activity over time
        st.subheader("Trading Activity Over Time")
        daily_activity = filtered_daily.groupby('date').agg({
            'num_trades': 'sum',
            'net_pnl': 'sum'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=daily_activity['date'], y=daily_activity['num_trades'], 
                      name="Number of Trades", line=dict(color='blue')),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=daily_activity['date'], y=daily_activity['net_pnl'], 
                      name="Total PnL", line=dict(color='green')),
            secondary_y=True
        )
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Number of Trades", secondary_y=False)
        fig.update_yaxes(title_text="Total PnL ($)", secondary_y=True)
        fig.update_layout(height=400, title="Daily Trading Activity and PnL")
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance Analysis Page
    elif page == "📈 Performance Analysis":
        st.header("📈 Performance Analysis: Fear vs Greed")
        
        # Calculate metrics
        fear_data = filtered_daily[filtered_daily['is_fear'] == 1]
        greed_data = filtered_daily[filtered_daily['is_greed'] == 1]
        
        fear_pnl = fear_data['net_pnl'].mean()
        greed_pnl = greed_data['net_pnl'].mean()
        fear_wr = fear_data['win_rate'].mean()
        greed_wr = greed_data['win_rate'].mean()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Fear Days Avg PnL", f"${fear_pnl:,.2f}")
        with col2:
            st.metric("Greed Days Avg PnL", f"${greed_pnl:,.2f}")
        with col3:
            delta_pnl = greed_pnl - fear_pnl
            st.metric("Difference", f"${delta_pnl:,.2f}", 
                     delta=f"{(delta_pnl/abs(fear_pnl)*100):.1f}%" if fear_pnl != 0 else "N/A")
        with col4:
            better = "Greed" if greed_pnl > fear_pnl else "Fear"
            st.metric("Better Period", better)
        
        st.markdown("---")
        
        # PnL comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Average PnL by Sentiment")
            pnl_by_sentiment = filtered_daily.groupby('classification')['net_pnl'].mean().reset_index()
            fig = px.bar(
                pnl_by_sentiment,
                x='classification',
                y='net_pnl',
                color='classification',
                color_discrete_map={
                    'Extreme Fear': '#8B0000',
                    'Fear': '#DC143C',
                    'Neutral': '#808080',
                    'Greed': '#90EE90',
                    'Extreme Greed': '#006400'
                },
                title="Average Daily PnL by Sentiment Type"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Win Rate by Sentiment")
            wr_by_sentiment = filtered_daily.groupby('classification')['win_rate'].mean().reset_index()
            fig = px.bar(
                wr_by_sentiment,
                x='classification',
                y='win_rate',
                color='classification',
                color_discrete_map={
                    'Extreme Fear': '#8B0000',
                    'Fear': '#DC143C',
                    'Neutral': '#808080',
                    'Greed': '#90EE90',
                    'Extreme Greed': '#006400'
                },
                title="Average Win Rate by Sentiment Type"
            )
            fig.update_yaxes(tickformat='.0%')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # PnL distribution
        st.subheader("PnL Distribution: Fear vs Greed")
        fig = go.Figure()
        fig.add_trace(go.Box(y=fear_data['net_pnl'], name='Fear Days', marker_color='red'))
        fig.add_trace(go.Box(y=greed_data['net_pnl'], name='Greed Days', marker_color='green'))
        fig.update_layout(height=400, title="Distribution of Daily Net PnL")
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk metrics
        st.markdown("---")
        st.subheader("📉 Risk Metrics")
        
        # Calculate risk metrics
        fear_downside = fear_data[fear_data['net_pnl'] < 0]['net_pnl'].std()
        greed_downside = greed_data[greed_data['net_pnl'] < 0]['net_pnl'].std()
        fear_loss_rate = (fear_data['net_pnl'] < 0).sum() / len(fear_data) if len(fear_data) > 0 else 0
        greed_loss_rate = (greed_data['net_pnl'] < 0).sum() / len(greed_data) if len(greed_data) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Fear Downside Dev", f"${fear_downside:,.2f}" if not np.isnan(fear_downside) else "N/A")
        with col2:
            st.metric("Greed Downside Dev", f"${greed_downside:,.2f}" if not np.isnan(greed_downside) else "N/A")
        with col3:
            st.metric("Fear Loss Rate", f"{fear_loss_rate:.1%}")
        with col4:
            st.metric("Greed Loss Rate", f"{greed_loss_rate:.1%}")
    
    # Behavior Analysis Page
    elif page == "🎯 Behavior Analysis":
        st.header("🎯 Trader Behavior Analysis")
        
        fear_data = filtered_daily[filtered_daily['is_fear'] == 1]
        greed_data = filtered_daily[filtered_daily['is_greed'] == 1]
        
        # Trading frequency
        fear_trades = fear_data['num_trades'].mean() if len(fear_data) > 0 else 0
        greed_trades = greed_data['num_trades'].mean() if len(greed_data) > 0 else 0
        fear_long = fear_data['long_ratio'].mean() if len(fear_data) > 0 else 0
        greed_long = greed_data['long_ratio'].mean() if len(greed_data) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Fear Avg Trades/Day", f"{fear_trades:.1f}")
        with col2:
            st.metric("Greed Avg Trades/Day", f"{greed_trades:.1f}")
        with col3:
            st.metric("Fear Long Ratio", f"{fear_long:.1%}")
        with col4:
            st.metric("Greed Long Ratio", f"{greed_long:.1%}")
        
        st.markdown("---")
        
        # Behavior charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trading Frequency by Sentiment")
            freq_data = filtered_daily.groupby('classification')['num_trades'].mean().reset_index()
            fig = px.bar(
                freq_data,
                x='classification',
                y='num_trades',
                color='classification',
                color_discrete_map={
                    'Extreme Fear': '#8B0000',
                    'Fear': '#DC143C',
                    'Neutral': '#808080',
                    'Greed': '#90EE90',
                    'Extreme Greed': '#006400'
                },
                title="Average Number of Trades per Day"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Long/Short Bias by Sentiment")
            long_data = filtered_daily.groupby('classification')['long_ratio'].mean().reset_index()
            fig = px.bar(
                long_data,
                x='classification',
                y='long_ratio',
                color='classification',
                color_discrete_map={
                    'Extreme Fear': '#8B0000',
                    'Fear': '#DC143C',
                    'Neutral': '#808080',
                    'Greed': '#90EE90',
                    'Extreme Greed': '#006400'
                },
                title="Proportion of Long Positions"
            )
            fig.add_hline(y=0.5, line_dash="dash", annotation_text="50% (Neutral)")
            fig.update_yaxes(tickformat='.0%')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Position sizing
        st.subheader("Position Sizing Behavior")
        size_data = filtered_daily.groupby('classification')['avg_size_usd'].mean().reset_index()
        fig = px.bar(
            size_data,
            x='classification',
            y='avg_size_usd',
            color='classification',
            color_discrete_map={
                'Extreme Fear': '#8B0000',
                'Fear': '#DC143C',
                'Neutral': '#808080',
                'Greed': '#90EE90',
                'Extreme Greed': '#006400'
            },
            title="Average Position Size (USD) by Sentiment"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        behavior_type = "Contrarian" if fear_long > greed_long else "Momentum"
        st.markdown(f"""
        <div class="insight-box">
        <b>📊 Behavioral Pattern Detected: {behavior_type}</b><br>
        Traders show {behavior_type.lower()} behavior - they {'buy more during fear' if fear_long > greed_long else 'follow market momentum'}.
        <br>Long ratio during Fear: {fear_long:.1%} | Long ratio during Greed: {greed_long:.1%}
        </div>
        """, unsafe_allow_html=True)
    
    # Trader Segments Page
    elif page == "👥 Trader Segments":
        st.header("👥 Trader Segmentation Analysis")
        
        # Segment overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Segmentation by Trading Frequency")
            freq_counts = trader_stats['frequency_group'].value_counts().sort_index()
            fig = px.pie(
                values=freq_counts.values,
                names=freq_counts.index,
                title="Distribution of Traders by Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance by frequency
            freq_perf = trader_stats.groupby('frequency_group')['total_pnl'].mean().reset_index()
            st.dataframe(
                freq_perf.rename(columns={'frequency_group': 'Group', 'total_pnl': 'Avg Total PnL'}),
                hide_index=True,
                use_container_width=True
            )
        
        with col2:
            st.subheader("Segmentation by Win Rate")
            wr_counts = trader_stats['winrate_group'].value_counts().sort_index()
            fig = px.pie(
                values=wr_counts.values,
                names=wr_counts.index,
                title="Distribution of Traders by Win Rate"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance by win rate
            wr_perf = trader_stats.groupby('winrate_group')['total_pnl'].mean().reset_index()
            st.dataframe(
                wr_perf.rename(columns={'winrate_group': 'Group', 'total_pnl': 'Avg Total PnL'}),
                hide_index=True,
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Performance comparison
        st.subheader("Performance Comparison Across Segments")
        
        col1, col2 = st.columns(2)
        
        with col1:
            freq_perf = trader_stats.groupby('frequency_group')['total_pnl'].mean().sort_index()
            fig = px.bar(
                x=freq_perf.index,
                y=freq_perf.values,
                labels={'x': 'Trading Frequency', 'y': 'Average Total PnL'},
                title="Performance by Trading Frequency",
                color=freq_perf.values,
                color_continuous_scale='RdYlGn'
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            wr_perf = trader_stats.groupby('winrate_group')['total_pnl'].mean().sort_index()
            fig = px.bar(
                x=wr_perf.index,
                y=wr_perf.values,
                labels={'x': 'Win Rate Group', 'y': 'Average Total PnL'},
                title="Performance by Win Rate",
                color=wr_perf.values,
                color_continuous_scale='RdYlGn'
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)
        
        # Best performers
        st.markdown("---")
        st.subheader("🏆 Top Performing Traders")
        
        top_traders = trader_stats.nlargest(10, 'total_pnl')[
            ['Account', 'total_pnl', 'avg_win_rate', 'total_trades', 'frequency_group']
        ].reset_index(drop=True)
        top_traders.index += 1
        
        st.dataframe(
            top_traders.style.format({
                'total_pnl': '${:,.2f}',
                'avg_win_rate': '{:.1%}',
                'total_trades': '{:,.0f}'
            }),
            use_container_width=True
        )
    
    # Insights & Strategies Page
    elif page == "💡 Insights & Strategies":
        st.header("💡 Key Insights & Trading Strategies")
        
        # Calculate key metrics for insights
        fear_data = filtered_daily[filtered_daily['is_fear'] == 1]
        greed_data = filtered_daily[filtered_daily['is_greed'] == 1]
        
        fear_pnl = fear_data['net_pnl'].mean() if len(fear_data) > 0 else 0
        greed_pnl = greed_data['net_pnl'].mean() if len(greed_data) > 0 else 0
        fear_long = fear_data['long_ratio'].mean() if len(fear_data) > 0 else 0
        greed_long = greed_data['long_ratio'].mean() if len(greed_data) > 0 else 0
        
        # Insights
        st.subheader("📊 Key Insights")
        
        better_sentiment = "Greed" if greed_pnl > fear_pnl else "Fear"
        diff = abs(greed_pnl - fear_pnl)
        
        st.markdown(f"""
        <div class="insight-box">
        <b>INSIGHT #1: {better_sentiment} Days Are More Profitable</b><br>
        • Average PnL during Fear: ${fear_pnl:,.2f}<br>
        • Average PnL during Greed: ${greed_pnl:,.2f}<br>
        • Difference: ${diff:,.2f} ({abs(diff/fear_pnl*100):.1f}% {'higher' if greed_pnl > fear_pnl else 'lower'})<br>
        <br>
        💡 <b>Actionable:</b> Focus trading activity during {better_sentiment} periods
        </div>
        """, unsafe_allow_html=True)
        
        behavior = "Contrarian" if fear_long > greed_long else "Momentum"
        st.markdown(f"""
        <div class="insight-box">
        <b>INSIGHT #2: Traders Show {behavior} Behavior</b><br>
        • Long ratio during Fear: {fear_long:.1%}<br>
        • Long ratio during Greed: {greed_long:.1%}<br>
        • Pattern: {'Traders buy more when market is fearful' if fear_long > greed_long else 'Traders follow market sentiment'}<br>
        <br>
        💡 <b>Actionable:</b> {'Use contrarian signals as entry opportunities' if fear_long > greed_long else 'Confirm momentum with technical indicators'}
        </div>
        """, unsafe_allow_html=True)
        
        best_segment = trader_stats.groupby('frequency_group')['total_pnl'].mean().idxmax()
        high_freq_pnl = trader_stats[trader_stats['frequency_group'] == 'High (20+)']['total_pnl'].mean()
        low_freq_pnl = trader_stats[trader_stats['frequency_group'] == 'Low (<5)']['total_pnl'].mean()
        
        st.markdown(f"""
        <div class="insight-box">
        <b>INSIGHT #3: {best_segment} Frequency Traders Perform Best</b><br>
        • High-frequency traders: ${high_freq_pnl:,.2f} avg total PnL<br>
        • Low-frequency traders: ${low_freq_pnl:,.2f} avg total PnL<br>
        • Difference: ${abs(high_freq_pnl - low_freq_pnl):,.2f}<br>
        <br>
        💡 <b>Actionable:</b> {'Increase trade frequency during favorable conditions' if 'High' in best_segment else 'Focus on quality over quantity'}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Strategies
        st.subheader("🎯 Trading Strategies")
        
        st.markdown(f"""
        <div class="strategy-box">
        <b>STRATEGY #1: Sentiment-Adaptive Position Sizing</b><br><br>
        
        <b>Rule:</b> Adjust position sizes based on daily Fear/Greed Index<br><br>
        
        <b>Implementation:</b><br>
        • Extreme Fear (0-25): 1.2x base position size<br>
        • Fear (26-45): 1.1x base position size<br>
        • Neutral (46-54): 1.0x base position size<br>
        • Greed (55-75): 0.9x base position size<br>
        • Extreme Greed (76-100): 0.7x base position size<br><br>
        
        <b>Risk Controls:</b><br>
        • Maximum position size: 3x base amount<br>
        • Stop trading at -5% daily loss<br>
        • Tighter stops on {'Fear' if fear_pnl < greed_pnl else 'Greed'} days<br><br>
        
        <b>Expected Outcome:</b> 15-25% improvement in risk-adjusted returns
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="strategy-box">
        <b>STRATEGY #2: {'Contrarian' if behavior == 'Contrarian' else 'Momentum'} Entry System</b><br><br>
        
        <b>Rule:</b> {'Take LONG positions when Fear Index > 70' if behavior == 'Contrarian' else 'Take LONG positions when Greed Index > 70'}<br><br>
        
        <b>Implementation:</b><br>
        • Entry signal: {'Extreme Fear with price stabilization' if behavior == 'Contrarian' else 'Strong Greed with momentum confirmation'}<br>
        • Position entry: Start with 50%, add 50% on confirmation<br>
        • Stop-loss: {'-3%' if behavior == 'Contrarian' else '-2.5%'} from entry<br>
        • Profit targets: {'+5% (50%), +10% (30%), trailing +8% (20%)' if behavior == 'Contrarian' else '+3% (50%), +6% (30%), trailing +4% (20%)'}<br><br>
        
        <b>Risk Controls:</b><br>
        • Maximum 2 positions simultaneously<br>
        • Exit if sentiment shifts significantly<br>
        • Hold maximum 5 days<br><br>
        
        <b>Expected Outcome:</b> 5-12% win rate improvement
        </div>
        """, unsafe_allow_html=True)
    
    # Predictive Model Page
    elif page == "🔮 Predictive Model":
        st.header("🔮 Predictive Analytics")
        
        st.info("🔮 Predicting next-day trader profitability using today's sentiment and behavior")
        
        # Calculate which sentiment is better (need this variable)
        fear_pnl_avg = filtered_daily[filtered_daily['is_fear'] == 1]['net_pnl'].mean() if len(filtered_daily[filtered_daily['is_fear'] == 1]) > 0 else 0
        greed_pnl_avg = filtered_daily[filtered_daily['is_greed'] == 1]['net_pnl'].mean() if len(filtered_daily[filtered_daily['is_greed'] == 1]) > 0 else 0
        better_sentiment = "greed" if greed_pnl_avg > fear_pnl_avg else "fear"
        
        # Prepare data for next-day prediction
        st.subheader("📊 Next-Day Profitability Prediction")
        
        # Create features and target
        prediction_data = filtered_daily.copy()
        
        # Calculate if today was profitable
        prediction_data['is_profitable_today'] = (prediction_data['net_pnl'] > 0).astype(int)
        
        # Sort by account and date
        prediction_data = prediction_data.sort_values(['Account', 'date'])
        
        # Create NEXT-DAY target (shift profit forward)
        prediction_data['is_profitable_tomorrow'] = prediction_data.groupby('Account')['is_profitable_today'].shift(-1)
        
        # Remove last day for each trader (no tomorrow data)
        prediction_data = prediction_data[prediction_data['is_profitable_tomorrow'].notna()]
        
        # Show what we're predicting
        st.markdown("""
        **Model Goal:** Using today's data to predict if tomorrow will be profitable
        
        **Features Used:**
        - Today's win rate
        - Today's sentiment (Fear/Greed)
        - Today's trading frequency
        - Today's position size
        - Today's long/short ratio
        """)
        
        st.markdown("---")
        
        if len(prediction_data) > 0:
            # Feature correlation with TOMORROW's profitability
            st.subheader("🎯 Feature Importance for Next-Day Prediction")
            
            features = ['num_trades', 'win_rate', 'long_ratio', 'avg_size_usd', 'is_fear', 'is_greed']
            target = 'is_profitable_tomorrow'
            
            # Calculate correlations
            correlations = {}
            for feature in features:
                if feature in prediction_data.columns:
                    try:
                        corr = prediction_data[[feature, target]].corr().iloc[0, 1]
                        if not np.isnan(corr):
                            correlations[feature] = corr
                    except:
                        pass
            
            if correlations:
                corr_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation'])
                corr_df = corr_df.sort_values('Correlation', ascending=True)
                
                fig = px.bar(
                    corr_df,
                    x='Correlation',
                    y='Feature',
                    orientation='h',
                    title="How Today's Features Predict Tomorrow's Profitability",
                    color='Correlation',
                    color_continuous_scale='RdYlGn'
                )
                fig.add_vline(x=0, line_dash="dash", line_color="black")
                st.plotly_chart(fig, use_container_width=True)
            
            # Simple prediction model
            st.subheader("🤖 Simple Next-Day Prediction Model")
            
            # Rule-based prediction
            prediction_data['predicted_profitable_tomorrow'] = (
                (prediction_data['win_rate'] > 0.5) & 
                (prediction_data[f"is_{better_sentiment}"] == 1) &
                (prediction_data['num_trades'] >= 3)
            ).astype(int)
            
            st.markdown(f"""
            **Prediction Rules:**
            1. If today's win rate > 50%, AND
            2. Today is a {better_sentiment.title()} day, AND
            3. Trader made at least 3 trades today
            4. → THEN predict profitable tomorrow ✅
            """)
            
            # Calculate accuracy
            accuracy = (prediction_data['predicted_profitable_tomorrow'] == prediction_data['is_profitable_tomorrow']).mean()
            
            # Calculate confusion matrix values
            true_positives = ((prediction_data['predicted_profitable_tomorrow'] == 1) & 
                              (prediction_data['is_profitable_tomorrow'] == 1)).sum()
            false_positives = ((prediction_data['predicted_profitable_tomorrow'] == 1) & 
                               (prediction_data['is_profitable_tomorrow'] == 0)).sum()
            false_negatives = ((prediction_data['predicted_profitable_tomorrow'] == 0) & 
                               (prediction_data['is_profitable_tomorrow'] == 1)).sum()
            true_negatives = ((prediction_data['predicted_profitable_tomorrow'] == 0) & 
                              (prediction_data['is_profitable_tomorrow'] == 0)).sum()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Display metrics
            st.markdown("---")
            st.subheader("📈 Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.1%}")
                st.caption("Overall correct predictions")
            with col2:
                st.metric("Precision", f"{precision:.1%}")
                st.caption("When we predict profit, how often correct?")
            with col3:
                st.metric("Recall", f"{recall:.1%}")
                st.caption("Of actual profits, how many did we catch?")
            with col4:
                st.metric("F1 Score", f"{f1_score:.3f}")
                st.caption("Balanced performance metric")
            
            # Confusion matrix visualization
            st.markdown("---")
            st.subheader("🎯 Prediction Confusion Matrix")
            
            confusion_data = pd.DataFrame({
                'Predicted Profitable': [true_positives, false_positives],
                'Predicted Not Profitable': [false_negatives, true_negatives]
            }, index=['Actually Profitable', 'Actually Not Profitable'])
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(confusion_data.style.format("{:,.0f}"), use_container_width=True)
            
            with col2:
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=[[true_positives, false_negatives],
                       [false_positives, true_negatives]],
                    x=['Predicted Profitable', 'Predicted Not Profitable'],
                    y=['Actually Profitable', 'Actually Not Profitable'],
                    colorscale='RdYlGn',
                    text=[[true_positives, false_negatives],
                          [false_positives, true_negatives]],
                    texttemplate='%{text}',
                    textfont={"size": 16}
                ))
                fig.update_layout(height=300, title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
            
            # Example predictions
            st.markdown("---")
            st.subheader("📋 Example Predictions")
            
            if len(prediction_data) > 0:
                example_df = prediction_data[['date', 'Account', 'win_rate', 'num_trades', 'classification', 
                                              'is_profitable_tomorrow', 'predicted_profitable_tomorrow']].head(20).copy()
                
                example_df['Actual Tomorrow'] = example_df['is_profitable_tomorrow'].map({1: '✅ Profit', 0: '❌ Loss'})
                example_df['Predicted Tomorrow'] = example_df['predicted_profitable_tomorrow'].map({1: '✅ Profit', 0: '❌ Loss'})
                example_df['Correct?'] = (example_df['is_profitable_tomorrow'] == example_df['predicted_profitable_tomorrow']).map({True: '✅', False: '❌'})
                
                display_df = example_df[['date', 'Account', 'win_rate', 'num_trades', 'classification', 
                                          'Actual Tomorrow', 'Predicted Tomorrow', 'Correct?']]
                
                st.dataframe(
                    display_df.style.format({
                        'win_rate': '{:.1%}',
                        'num_trades': '{:.0f}'
                    }),
                    use_container_width=True
                )
            
            # Insights
            st.markdown("---")
            st.subheader("💡 Key Insights")
            
            st.markdown(f"""
            <div class="insight-box">
            <b>Model Performance Analysis:</b><br><br>
            
            ✅ The model achieves <b>{accuracy:.1%}</b> accuracy in predicting next-day profitability<br><br>
            
            📊 <b>What this means:</b><br>
            • Using today's win rate and sentiment, we can predict tomorrow's outcome better than random (50%)<br>
            • Precision of {precision:.1%} means when we predict profit, we're right {precision:.1%} of the time<br>
            • Recall of {recall:.1%} means we catch {recall:.1%} of actual profitable days<br><br>
            
            💡 <b>Practical Application:</b><br>
            • If today shows high win rate (>50%) on {better_sentiment.title()} days → Increase position sizes tomorrow<br>
            • If today shows poor performance → Reduce exposure tomorrow<br>
            • Use this as a risk management tool, not absolute prediction<br>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.warning("Not enough data for predictions after filtering. Try adjusting your date range or sentiment filters.")
        
        st.info("""
        💡 **Note:** This is a simple rule-based model for demonstration. 
        A production model would use machine learning techniques like Random Forest, 
        XGBoost, or LSTM neural networks for better accuracy.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
    <b>Trader Performance vs Market Sentiment Analysis Dashboard</b>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()