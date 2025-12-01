import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function definitions from previous steps
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Handle missing values as per previous steps
    df.dropna(subset=['artist_name'], inplace=True)
    df['artist_genres'] = df['artist_genres'].fillna('unknown')

    # Convert 'album_release_date' to datetime object
    df['album_release_date'] = pd.to_datetime(df['album_release_date'])

    # Feature Engineering: 'duration_in_minutes'
    df['duration_in_minutes'] = df['track_duration_min']

    # Feature Engineering: 'popularity_category'
    bins = [0, 25, 50, 75, 100]
    labels = ['Low', 'Medium', 'High', 'Very High']
    df['popularity_category'] = pd.cut(df['track_popularity'], bins=bins, labels=labels, right=False)

    # Feature Engineering: 'release_year'
    df['release_year'] = df['album_release_date'].dt.year

    return df

def get_top_artists(df, n=10):
    """Returns the top N artists based on their average track popularity."""
    top_artists = df.groupby('artist_name')['track_popularity'].mean().sort_values(ascending=False).head(n)
    return top_artists

def get_popularity_by_genre(df):
    """Calculates the average track popularity for each artist genre."""
    df_genres_exploded = df.assign(artist_genres=df['artist_genres'].str.split(', ')).explode('artist_genres')
    genre_popularity = df_genres_exploded.groupby('artist_genres')['track_popularity'].mean().sort_values(ascending=False)
    return genre_popularity

def get_correlation_matrix(df):
    """Calculates the correlation matrix for specified numerical features."""
    numerical_cols = ['track_popularity', 'artist_popularity', 'artist_followers', 'track_duration_min', 'album_total_tracks']
    correlation_matrix = df[numerical_cols].corr()
    return correlation_matrix

# Streamlit App
st.set_page_config(layout="wide")
st.title('Spotify Data Analysis Dashboard')

# File path for the dataset
file_path = '/content/spotify_data_clean.csv'

# Load and preprocess data
with st.spinner('Loading and preprocessing data...'):
    processed_df = load_and_preprocess_data(file_path)
st.success('Data loaded and preprocessed successfully!')

# Display Processed Data
st.header('Processed Data Overview')
st.write('First 5 rows of the processed DataFrame:')
st.dataframe(processed_df.head())
st.write(f"DataFrame shape: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")

# Key Insights
st.header('Key Insights')

# Top Artists by Popularity
st.subheader('Top 10 Artists by Average Track Popularity')
top_artists_data = get_top_artists(processed_df, n=10)
st.dataframe(top_artists_data)

# Top Genres by Popularity
st.subheader('Top 10 Genres by Average Track Popularity')
genre_popularity_data = get_popularity_by_genre(processed_df).head(10)
st.dataframe(genre_popularity_data)

# Visualizations
st.header('Data Visualizations')

# 1. Distribution of 'popularity_category'
st.subheader('Distribution of Track Popularity Categories')
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.countplot(x='popularity_category', data=processed_df, order=['Low', 'Medium', 'High', 'Very High'], hue='popularity_category', palette='viridis', ax=ax1)
ax1.set_title('Distribution of Track Popularity Categories')
ax1.set_xlabel('Popularity Category')
ax1.set_ylabel('Number of Tracks')
st.pyplot(fig1)
plt.clf()

# 2. Distribution of 'track_popularity'
st.subheader('Distribution of Track Popularity')
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.histplot(processed_df['track_popularity'], bins=20, kde=True, color='skyblue', ax=ax2)
ax2.set_title('Distribution of Track Popularity')
ax2.set_xlabel('Track Popularity')
ax2.set_ylabel('Frequency')
st.pyplot(fig2)
plt.clf()

# 3. Correlations between numerical features
st.subheader('Correlation Matrix of Numerical Features')
correlation_data = get_correlation_matrix(processed_df)
fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax3)
ax3.set_title('Correlation Matrix of Numerical Features')
st.pyplot(fig3)
plt.clf()

# 4. Track trends over release year
st.subheader('Average Track Popularity Over Release Years')
popularity_by_year = processed_df.groupby('release_year')['track_popularity'].mean().reset_index()
fig4, ax4 = plt.subplots(figsize=(12, 7))
sns.lineplot(x='release_year', y='track_popularity', data=popularity_by_year, marker='o', color='green', label='Average Track Popularity', ax=ax4)
ax4.set_title('Average Track Popularity Over Release Years')
ax4.set_xlabel('Release Year')
ax4.set_ylabel('Average Track Popularity')
ax4.grid(True)
ax4.legend()
st.pyplot(fig4)
plt.clf()
