import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import networkx as nx

# =================================
# streamlit run Olympics_app.py
# =================================
# Data source: https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results
# Dataset files needed:
# - athlete_events.csv
# - noc_regions.csv

# If you don't have preprocess.py, replace this with your own preprocessing function
try:
    from preprocessor import preprocess
except Exception:
    def preprocess(df, regions_df=None):
        # Minimal preprocessing fallback: rename columns if needed and add `region`
        df = df.copy()
        # ensure common columns exist
        if 'NOC' in df.columns and (regions_df is not None):
            try:
                regions_df = regions_df.rename(columns={
                    'NOC': 'NOC', 'region': 'region'
                })
                df = df.merge(regions_df, how='left', left_on='NOC', right_on='NOC')
                if 'region' not in df.columns:
                    df['region'] = df.get('Team', df.get('NOC', None))
            except Exception:
                df['region'] = df.get('Team', df.get('NOC', None))
        else:
            df['region'] = df.get('Team', df.get('NOC', None))
        # add ID column if missing
        if 'ID' not in df.columns:
            df['ID'] = df.index
        return df

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="🏅 Olympics Dashboard", page_icon="🥇", layout="wide")

# -------------------------
# Data loader
# -------------------------
@st.cache_data(show_spinner=True)
def load_data(events_path=None, regions_path=None):
    try:
        if events_path is None:
            events_path = r"C:\Users\Hp\Downloads\athlete_events.csv\athlete_events.csv"
        df = pd.read_csv(events_path)
        region_df = None
        if regions_path is None:
            try:
                regions_path = r"C:\Users\Hp\Downloads\noc_regions.csv"
                region_df = pd.read_csv(regions_path)
            except Exception:
                region_df = None
        df = preprocess(df, region_df)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# Load
df = load_data()
if df is None:
    st.stop()

# Ensure some helpful column types and cleanups
# Common column name fixes
col_map = {}
if 'Medal' not in df.columns and 'medal' in df.columns:
    col_map['medal'] = 'Medal'
if 'Team' in df.columns and 'region' not in df.columns:
    col_map['Team'] = 'region'
if col_map:
    df = df.rename(columns=col_map)

# Fill basic missing columns with safe defaults
for c in ['Age', 'Height', 'Weight', 'Sport', 'Year', 'Event', 'Name', 'ID', 'Medal', 'region']:
    if c not in df.columns:
        df[c] = np.nan

# Convert Year to numeric if possible
try:
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
except Exception:
    pass

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("🏅 Olympics Dashboard")
menu = st.sidebar.radio("Select Section", (
    "🏆 Medal Tally",
    "📈 Overall Analysis",
    "🌍 Country Analysis",
    "👨‍🦱 Athlete Analysis",
    "🤖 AI Insights",
    "🔬 Advanced Analytics",
    "🌐 Network Analysis"
))

# -------------------------
# Helper utils
# -------------------------
def safe_unique_sorted(series):
    return sorted([x for x in series.dropna().unique()]) if series is not None else []

# -------------------------
# 1) Medal Tally
# -------------------------
if menu == "🏆 Medal Tally":
    st.header("🏆 Medal Tally")
    if 'region' not in df.columns or 'Medal' not in df.columns:
        st.info("Dataset missing 'region' or 'Medal' column. Check preprocessing.")
    else:
        year_options = [int(y) for y in df['Year'].dropna().unique() if not pd.isna(y)]
        year_options = sorted(year_options)
        year_sel = st.selectbox("Select Year (or All)", ["All"] + year_options)
        sport_sel = st.selectbox("Filter by Sport", ["All"] + safe_unique_sorted(df['Sport']))

        data = df.copy()
        if year_sel != "All":
            data = data[data['Year'] == int(year_sel)]
        if sport_sel != "All":
            data = data[data['Sport'] == sport_sel]

        # compute medal counts
        medal_df = pd.crosstab(data['region'], data['Medal']).fillna(0)
        # Some datasets put medals as strings 'Gold','Silver','Bronze' - ensure columns exist
        for m in ['Gold', 'Silver', 'Bronze']:
            if m not in medal_df.columns:
                medal_df[m] = 0
        medal_df['Total'] = medal_df[['Gold', 'Silver', 'Bronze']].sum(axis=1)
        medal_df = medal_df.sort_values('Gold', ascending=False)

        st.subheader("Medal Table")
        st.dataframe(medal_df.reset_index().rename(columns={'index': 'Country'}), use_container_width=True)

        st.subheader("Top countries (by Total Medals)")
        top = medal_df.sort_values('Total', ascending=False).head(15)
        fig = px.bar(top.reset_index(), x='Total', y=top.reset_index()['region'] if 'region' in top.reset_index().columns else top.reset_index()['index'], orientation='h', labels={'y': 'Country'})
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 2) Overall Analysis
# -------------------------
elif menu == "📈 Overall Analysis":
    st.header("📈 Overall Analysis")
    cols = st.columns(4)
    # safe counts
    total_olympics = df['Year'].nunique(dropna=True)
    total_countries = df['region'].nunique(dropna=True)
    total_athletes = df['ID'].nunique(dropna=True)
    total_events = df['Event'].nunique(dropna=True)

    cols[0].metric("Olympics (years)", int(total_olympics))
    cols[1].metric("Countries", int(total_countries))
    cols[2].metric("Unique Athletes", int(total_athletes))
    cols[3].metric("Events", int(total_events))

    st.subheader("Athletes & Events over Years")
    if 'Year' in df.columns:
        yearly = df.groupby('Year').agg({'ID': 'nunique', 'Event': 'nunique'}).reset_index().dropna()
        if not yearly.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['ID'], mode='lines+markers', name='Athletes'))
            fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['Event'], mode='lines+markers', name='Events'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No yearly data available')
    else:
        st.info('Year column missing')

# -------------------------
# 3) Country Analysis
# -------------------------
elif menu == "🌍 Country Analysis":
    st.header("🌍 Country Analysis")
    if 'region' not in df.columns:
        st.info("No region column present")
    else:
        country = st.selectbox("Select Country", [None] + safe_unique_sorted(df['region']))
        if country:
            cdata = df[df['region'] == country]
            st.subheader(f"Overview: {country}")
            medals_by_year = cdata.groupby('Year')['Medal'].apply(lambda x: x.notna().sum()).reset_index()
            if not medals_by_year.empty:
                st.line_chart(medals_by_year.set_index('Year')['Medal'])
            else:
                st.info('No medal history for this country')

            st.subheader('Top Athletes')
            if 'Name' in cdata.columns:
                top_ath = cdata.groupby('Name')['Medal'].apply(lambda x: x.notna().sum()).sort_values(ascending=False).head(20)
                st.dataframe(top_ath.reset_index().rename(columns={'Name': 'Athlete', 'Medal': 'Medals'}), use_container_width=True)
            else:
                st.info('Name column missing')
            st.subheader('Sport-wise Medal Distribution')
            if 'Sport' in cdata.columns and 'Medal' in cdata.columns:
                sport_medals = cdata.groupby('Sport')['Medal'].apply(lambda x: x.notna().sum()).sort_values(ascending=False).head(10)
                fig = px.bar(sport_medals.reset_index(), x='Sport', y='Medal', labels={'Medal': 'Total Medals'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('Sport or Medal column missing')
                

# -------------------------
# 4) Athlete Analysis
# -------------------------
elif menu == "👨‍🦱 Athlete Analysis":
    st.header("👨‍🦱 Athlete Analysis")
    if 'Name' not in df.columns:
        st.info('No Name column')
    else:
        athlete = st.selectbox('Select Athlete', [None] + safe_unique_sorted(df['Name']))
        if athlete:
            adata = df[df['Name'] == athlete]
            st.subheader(f'Profile: {athlete}')
            st.write(adata[['Year', 'Sport', 'Event', 'Medal']].drop_duplicates().sort_values('Year'))

        st.subheader('Top Medalists')
        top_med = df.groupby('Name')['Medal'].apply(lambda x: x.notna().sum()).sort_values(ascending=False).head(30)
        st.dataframe(top_med.reset_index().rename(columns={'Name': 'Athlete', 'Medal': 'Medals'}), use_container_width=True)

        if set(['Age', 'Height', 'Weight']).issubset(df.columns):
            st.subheader('Physical Distributions')
            num = df[['Age', 'Height', 'Weight']].dropna()
            if not num.empty:
                fig = make_subplots(rows=1, cols=3, subplot_titles=('Age', 'Height', 'Weight'))
                fig.add_trace(go.Histogram(x=num['Age']), row=1, col=1)
                fig.add_trace(go.Histogram(x=num['Height']), row=1, col=2)
                fig.add_trace(go.Histogram(x=num['Weight']), row=1, col=3)
                st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 5) AI Insights (kept compact & safe)
# -------------------------
elif menu == "🤖 AI Insights":
    st.header('🤖 AI Insights')
    tab1, tab2, tab3 = st.tabs(['Clustering', 'Trends', 'Medal Prob'])
    # Clustering (safe)
    with tab1:
        numeric_cols = [c for c in ['Age', 'Height', 'Weight'] if c in df.columns]
        if len(numeric_cols) >= 2:
            cdata = df[numeric_cols + ['Sport', 'Medal']].dropna()
            if not cdata.empty:
                scaler = StandardScaler(); X = scaler.fit_transform(cdata[numeric_cols])
                k = st.slider('Clusters', 2, 8, 4)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cdata['Cluster'] = kmeans.fit_predict(X)
                st.dataframe(cdata.groupby('Cluster')[numeric_cols].mean().round(2))
            else:
                st.info('Not enough data to cluster')
        else:
            st.info('Need at least two numeric cols')
    with tab2:
        if 'Year' in df.columns:
            yearly = df.groupby('Year').agg({'ID': 'nunique'}).reset_index().dropna()
            if not yearly.empty:
                X = yearly['Year'].values.reshape(-1,1); y = yearly['ID'].values
                model = LinearRegression().fit(X,y)
                fut = np.array([2028,2032,2036]).reshape(-1,1)
                preds = model.predict(fut)
                fig = go.Figure(); fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['ID'], name='Actual'))
                fig.add_trace(go.Scatter(x=fut.flatten(), y=preds, name='Pred'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('No yearly stats')
    with tab3:
        st.info('Use Country Analysis or Athlete Analysis to inspect medal probabilities (simple historical lookups).')

# -------------------------
# 6) Advanced Analytics
# -------------------------
elif menu == "🔬 Advanced Analytics":
    st.header('🔬 Advanced Analytics')
    tab1, tab2, tab3 = st.tabs(['Efficiency','Correlation','Dominance'])
    with tab1:
        if set(['region','Medal','ID']).issubset(df.columns):
            stats = df.groupby('region').agg({'Medal': lambda x: x.notna().sum(),'ID':'nunique'}).reset_index()
            stats['Medal_per_Athlete'] = stats['Medal'] / stats['ID']
            st.dataframe(stats.sort_values('Medal_per_Athlete', ascending=False).head(20))
    with tab2:
        if set(['region','Sport','Medal']).issubset(df.columns):
            mat = df.pivot_table(values='Medal', index='region', columns='Sport', aggfunc=lambda x: x.notna().sum(), fill_value=0)
            if not mat.empty:
                st.plotly_chart(px.imshow(mat.corr()), use_container_width=True)
    with tab3:
        if set(['region','Sport','Medal']).issubset(df.columns):
            mc = df.groupby('region')['Medal'].apply(lambda x: x.notna().sum())
            st.bar_chart(mc.nlargest(15))

# -------------------------
# 7) Network Analysis (fixed)
# -------------------------
elif menu == "🌐 Network Analysis":
    st.header('🌐 Network Analysis')
    tab1, tab2 = st.tabs(['Country Collaboration','Athlete Connections'])
    with tab1:
        if set(['Year','region','Sport']).issubset(df.columns):
            years = sorted(df['Year'].dropna().unique(), reverse=True)
            if years:
                ysel = st.selectbox('Year', [years[0]] + years)
                year_data = df[df['Year'] == ysel].dropna(subset=['region','Sport'])
                if not year_data.empty:
                    sport_countries = year_data.groupby('Sport')['region'].apply(lambda s: list(s.dropna().unique())).to_dict()
                    connections = {}
                    for sport, countries in sport_countries.items():
                        for i, c1 in enumerate(countries):
                            for c2 in countries[i+1:]:
                                if c1 and c2:
                                    key = tuple(sorted([c1,c2]))
                                    connections[key] = connections.get(key,0) + 1
                    if connections:
                        top = sorted(connections.items(), key=lambda x: x[1], reverse=True)[:50]
                        dfc = pd.DataFrame([{'Country1':k[0][0],'Country2':k[0][1],'SharedSports':k[1]} for k in top])
                        st.dataframe(dfc)
                    else:
                        st.info('No collaborations found')
                else:
                    st.info('No data for selected year')
            else:
                st.info('No year information')
    with tab2:
        if set(['Name','Year']).issubset(df.columns):
            athlete_years = df.groupby('Name')['Year'].nunique()
            multi = athlete_years[athlete_years >= 3].index
            if len(multi) > 0:
                elite = df[df['Name'].isin(multi)]
                elite_summary = elite.groupby('Name').agg({'Year': lambda x: f"{int(x.min())}-{int(x.max())}", 'Medal': lambda x: x.notna().sum(), 'region':'first'}).sort_values('Medal', ascending=False).head(50)
                elite_summary.columns = ['OlympicSpan','TotalMedals','Country']
                st.dataframe(elite_summary)
            else:
                st.info('No multi-Olympic athletes found')

# Footer
st.markdown('---')
st.caption('This dashboard is a resilient, defensive-version intended to run even when some columns are missing. Adjust file paths and the preprocess() function as needed.')
