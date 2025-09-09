import pandas as pd

def _dedup_medals(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate team entries so team medals don't get overcounted."""
    temp = df.dropna(subset=['Medal']).copy()
    keys = ['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal', 'region']
    # Keep only columns that exist in df (to be robust)
    keys = [k for k in keys if k in temp.columns]
    temp = temp.drop_duplicates(subset=keys)
    return temp

def medal_tally(df: pd.DataFrame, year, country) -> pd.DataFrame:
    temp = _dedup_medals(df)

    if year != 'Overall':
        temp = temp[temp['Year'] == year]
    if country != 'Overall':
        temp = temp[temp['region'] == country]

    pivot = (temp.pivot_table(index='region', columns='Medal', values='ID', aggfunc='count')
                 .fillna(0)
                 .rename_axis(None, axis=1))

    for col in ['Gold', 'Silver', 'Bronze']:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot = pivot[['Gold', 'Silver', 'Bronze']]
    pivot['Total'] = pivot['Gold'] + pivot['Silver'] + pivot['Bronze']

    out = pivot.sort_values(['Gold', 'Silver', 'Bronze'], ascending=False).reset_index()
    return out

def country_year_list(df: pd.DataFrame):
    years = sorted(df['Year'].unique().tolist())
    years.insert(0, 'Overall')

    countries = sorted(df['region'].dropna().unique().tolist())
    countries.insert(0, 'Overall')

    return years, countries

def data_over_time(df: pd.DataFrame, col: str) -> pd.DataFrame:
    t = (df.drop_duplicates(['Year', col])
         .groupby('Year')
         .size()
         .reset_index(name=col))
    return t

def yearwise_medal_tally(df: pd.DataFrame, country: str) -> pd.DataFrame:
    temp = _dedup_medals(df)
    temp = temp[temp['region'] == country]
    final_df = (temp.groupby('Year')['Medal']
                    .count()
                    .reset_index(name='Medals'))
    return final_df

def country_event_heatmap(df: pd.DataFrame, country: str) -> pd.DataFrame:
    temp = _dedup_medals(df)
    temp = temp[temp['region'] == country]
    if temp.empty:
        return pd.DataFrame()
    pt = (temp.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count')
             .fillna(0).astype(int))
    return pt

def sports_event_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.drop_duplicates(['Year', 'Sport', 'Event'])
    pt = (temp.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count')
             .fillna(0).astype(int))
    return pt

def most_successful(df: pd.DataFrame, sport: str = 'Overall', country: str = None) -> pd.DataFrame:
    temp = df.dropna(subset=['Medal']).copy()
    if sport != 'Overall':
        temp = temp[temp['Sport'] == sport]
    if country:
        temp = temp[temp['region'] == country]

    rank = (temp.groupby(['Name', 'region', 'Sport'])['Medal']
                 .count()
                 .reset_index(name='Medals')
                 .sort_values('Medals', ascending=False))
    return rank

def age_distributions(df: pd.DataFrame) -> dict:
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    overall = athlete_df['Age'].dropna().astype(int).value_counts().sort_index()

    gold = df[df['Medal'] == 'Gold'].drop_duplicates(subset=['Name', 'region'])
    silver = df[df['Medal'] == 'Silver'].drop_duplicates(subset=['Name', 'region'])
    bronze = df[df['Medal'] == 'Bronze'].drop_duplicates(subset=['Name', 'region'])

    gold_counts = gold['Age'].dropna().astype(int).value_counts().sort_index()
    silver_counts = silver['Age'].dropna().astype(int).value_counts().sort_index()
    bronze_counts = bronze['Age'].dropna().astype(int).value_counts().sort_index()

    return {
        'overall': overall,
        'gold': gold_counts,
        'silver': silver_counts,
        'bronze': bronze_counts,
    }

def male_female_trend(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.drop_duplicates(['Year', 'Name'])
    trend = (temp.groupby(['Year', 'Sex'])['ID']
                  .count()
                  .reset_index(name='Count'))
    pt = trend.pivot(index='Year', columns='Sex', values='Count').fillna(0).astype(int).reset_index()
    return pt

def height_weight_data(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    cols = ['Height', 'Weight', 'Sex', 'Sport', 'Medal']
    temp = df.dropna(subset=['Height', 'Weight'])[cols].copy()
    if sport != 'Overall':
        temp = temp[temp['Sport'] == sport]

    temp['Medal'] = temp['Medal'].fillna('No Medal')
    return temp
