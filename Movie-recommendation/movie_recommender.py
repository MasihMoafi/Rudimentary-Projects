import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Generate sample movie data
def generate_sample_movies():
    np.random.seed(42)
    
    # Movie genres with weights
    genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
    
    # Generate 100 sample movies
    n_movies = 100
    movie_ids = np.arange(1, n_movies + 1)
    titles = [f"Movie {i}" for i in range(1, n_movies + 1)]
    
    # Generate random genre vectors (some movies have multiple genres)
    genre_vectors = np.zeros((n_movies, len(genres)))
    for i in range(n_movies):
        # Each movie has 1-3 genres
        num_genres = np.random.randint(1, 4)
        genre_indices = np.random.choice(len(genres), size=num_genres, replace=False)
        genre_vectors[i, genre_indices] = 1
    
    # Generate other features
    years = np.random.randint(1980, 2023, size=n_movies)
    lengths = np.random.randint(70, 180, size=n_movies)
    ratings = np.round(np.random.normal(7, 1.5, size=n_movies), 1)
    ratings = np.clip(ratings, 1, 10)  # Clip ratings to 1-10 range
    
    # Create DataFrame
    movies_df = pd.DataFrame({
        'movie_id': movie_ids,
        'title': titles,
        'year': years,
        'length': lengths,
        'rating': ratings
    })
    
    # Add genre columns
    for i, genre in enumerate(genres):
        movies_df[genre] = genre_vectors[:, i]
    
    return movies_df

# Generate sample user ratings
def generate_sample_ratings(movies_df, n_users=20):
    np.random.seed(42)
    
    user_ids = np.arange(1, n_users + 1)
    ratings = []
    
    for user_id in user_ids:
        # Each user rates 10-30 movies
        n_ratings = np.random.randint(10, 31)
        rated_movies = np.random.choice(movies_df['movie_id'], size=n_ratings, replace=False)
        
        # Generate ratings with some consistency
        # Users have preferences for certain genres
        preferred_genres = np.random.choice(
            ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller'], 
            size=np.random.randint(1, 4), 
            replace=False
        )
        
        for movie_id in rated_movies:
            movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
            
            # Base rating
            rating = np.random.normal(6, 1)
            
            # Boost rating for preferred genres
            for genre in preferred_genres:
                if movie[genre] == 1:
                    rating += np.random.uniform(0.5, 2)
            
            # Adjust rating based on movie's overall rating
            rating += (movie['rating'] - 5) * 0.3
            
            # Add some noise
            rating += np.random.normal(0, 0.5)
            
            # Clip rating to valid range and round
            rating = np.clip(round(rating, 1), 1, 10)
            
            ratings.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating
            })
    
    return pd.DataFrame(ratings)

# Content-based recommendation
def content_based_recommendation(movies_df, movie_id, n_recommendations=5):
    # Get the movie features
    feature_cols = ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'year', 'length', 'rating']
    
    # Create feature matrix
    X = movies_df[feature_cols].copy()
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_cols = ['year', 'length', 'rating']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Calculate similarity
    similarity = cosine_similarity(X)
    
    # Get the index of the movie
    movie_idx = movies_df[movies_df['movie_id'] == movie_id].index[0]
    
    # Get similarity scores
    movie_similarities = similarity[movie_idx]
    
    # Get top similar movies (excluding the movie itself)
    similar_movie_indices = np.argsort(movie_similarities)[::-1][1:n_recommendations+1]
    
    # Get recommended movies
    recommendations = movies_df.iloc[similar_movie_indices][['movie_id', 'title', 'year', 'rating']]
    
    return recommendations, movie_similarities[similar_movie_indices]

# Cluster-based recommendation
def cluster_based_recommendation(movies_df, ratings_df, user_id, n_recommendations=5):
    # Get user's rated movies
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    if len(user_ratings) == 0:
        print(f"No ratings found for user {user_id}")
        return None, None
    
    # Get feature matrix
    feature_cols = ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'year', 'length', 'rating']
    X = movies_df[feature_cols].copy()
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_cols = ['year', 'length', 'rating']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Apply KMeans clustering
    k = 5  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    movies_df['cluster'] = kmeans.fit_predict(X)
    
    # Find highly rated movies by the user
    highly_rated = user_ratings[user_ratings['rating'] >= 7.0]
    
    if len(highly_rated) == 0:
        print(f"No highly rated movies found for user {user_id}")
        return None, None
    
    # Get the clusters of highly rated movies
    rated_movie_ids = highly_rated['movie_id'].values
    rated_clusters = movies_df[movies_df['movie_id'].isin(rated_movie_ids)]['cluster'].values
    
    # Count frequency of each cluster
    cluster_counts = np.bincount(rated_clusters)
    favorite_cluster = np.argmax(cluster_counts)
    
    # Find unwatched movies from the favorite cluster
    watched_movies = user_ratings['movie_id'].values
    recommendations = movies_df[
        (movies_df['cluster'] == favorite_cluster) & 
        (~movies_df['movie_id'].isin(watched_movies))
    ].sort_values('rating', ascending=False).head(n_recommendations)
    
    if len(recommendations) == 0:
        print(f"No unwatched movies found in the favorite cluster for user {user_id}")
        return None, None
    
    return recommendations[['movie_id', 'title', 'year', 'rating']], favorite_cluster

# Visualize movie clusters
def visualize_clusters(movies_df):
    # Apply PCA to reduce to 2 dimensions for visualization
    from sklearn.decomposition import PCA
    
    feature_cols = ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'year', 'length', 'rating']
    X = movies_df[feature_cols].copy()
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_cols = ['year', 'length', 'rating']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Apply KMeans clustering
    k = 5  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    movies_df['cluster'] = kmeans.fit_predict(X)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i in range(k):
        plt.scatter(
            X_pca[movies_df['cluster'] == i, 0],
            X_pca[movies_df['cluster'] == i, 1],
            c=colors[i % len(colors)],
            label=f'Cluster {i}'
        )
    
    plt.title('Movie Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig('movie_clusters.png')
    plt.close()

def main():
    print("Generating sample movie data...")
    movies_df = generate_sample_movies()
    
    print("Generating sample user ratings...")
    ratings_df = generate_sample_ratings(movies_df)
    
    # Content-based recommendation
    print("\nContent-based recommendation example:")
    movie_id = 42  # Example movie ID
    movie_title = movies_df[movies_df['movie_id'] == movie_id]['title'].iloc[0]
    print(f"Finding movies similar to '{movie_title}'")
    
    recommendations, similarities = content_based_recommendation(movies_df, movie_id)
    print("\nRecommended movies:")
    for i, (_, row) in enumerate(recommendations.iterrows()):
        print(f"{i+1}. {row['title']} ({row['year']}) - Rating: {row['rating']}")
    
    # Cluster-based recommendation
    print("\nCluster-based recommendation example:")
    user_id = 5  # Example user ID
    print(f"Finding recommendations for user {user_id}")
    
    recommendations, cluster = cluster_based_recommendation(movies_df, ratings_df, user_id)
    if recommendations is not None:
        print(f"\nRecommended movies from cluster {cluster}:")
        for i, (_, row) in enumerate(recommendations.iterrows()):
            print(f"{i+1}. {row['title']} ({row['year']}) - Rating: {row['rating']}")
    
    # Visualize clusters
    print("\nVisualizing movie clusters...")
    visualize_clusters(movies_df)
    print("Visualization saved as 'movie_clusters.png'")

if __name__ == "__main__":
    main() 