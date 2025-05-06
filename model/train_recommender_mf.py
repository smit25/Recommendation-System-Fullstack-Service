import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

df = pd.read_csv("../movie_logs.csv")

# Create a unique movie identifier by combining movie name and year released
df['movie_id'] = df['movie_name'] + ' (' + df['year_released'].astype(str) + ')'

# Drop rows without ratings (the model will be trained only on observed ratings)
df_ratings = df.dropna(subset=['rating'])

# The Surprise library expects a DataFrame with columns: user, item, rating
# We will use 'user_id', 'movie_id', and 'rating'
reader = Reader(rating_scale=(df_ratings['rating'].min(), df_ratings['rating'].max()))
data_surprise = Dataset.load_from_df(df_ratings[['user_id', 'movie_id', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data_surprise, test_size=0.2, random_state=42)

# Use SVD for matrix factorization
algo = SVD(n_factors=20, n_epochs=20, random_state=42)
algo.fit(trainset)

# Evaluate the algorithm on the testset
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions)

print("Test RMSE: ", rmse)

# Example: Generate top-N recommendations for a given user (e.g., user_id=102833)
def get_top_n_recommendations(algo, user_id, df, n=5):
    # Get the list of all unique movies
    all_movie_ids = df['movie_id'].unique()
    
    # Movies already rated by the user
    rated_movies = df[df['user_id'] == user_id]['movie_id'].tolist()
    
    # Predict ratings for movies not yet rated by the user
    predictions = []
    for movie_id in all_movie_ids:
        if movie_id not in rated_movies:
            pred = algo.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))
    
    # Sort predictions by estimated rating in descending order and return the top-n movies
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

top_recommendations = get_top_n_recommendations(algo, user_id=102833, df=df)
print("Top recommendations for user 102833:")
for movie, rating in top_recommendations:
    print(f"{movie}: Predicted rating {rating:.2f}")