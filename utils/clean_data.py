import numpy as np
import pandas as pd
from utils.sql import get_ratings_df


def clean_data(df: pd.DataFrame=None, to_numpy=False) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], list[int], list[str]]:
    """
    Given the path to a .csv file for our data, returns a tuple of the following:
    (
        if to_numpy:
            (X_train, X_test_1, X_test_2): In each, rows corresponding to users, columns to movies,
                and values to ratings. Split is 80%, 10%, 10%
        else:
            (df_train, df_test_1, df_test_2): Long-form with 'user_id', 'movie_id', 'rating'
        user_list: List of user ids, index corresponding to row in ratings_mtx
        movie_list: List of movie names, index corresponding to column in ratings_mtx
    )
    """

    # df = pd.read_csv(path)
    if df is None:
        df = get_ratings_df()

    user_list = sorted(df['user_id'].unique())
    movie_list = sorted(df['movie_id'].unique())
    
    df_train, df_test_1, df_test_2 = train_test_split(df)

    if not to_numpy:
        df_train = df_train[['timestamp', 'user_id', 'movie_id', 'rating']]
        df_test_1 = df_test_1[['timestamp', 'user_id', 'movie_id', 'rating']]
        df_test_2 = df_test_2[['timestamp', 'user_id', 'movie_id', 'rating']]
        return (df_train, df_test_1, df_test_2), user_list, movie_list

    user_map = {v: k for k, v in enumerate(user_list)}
    movie_map = {v: k for k, v in enumerate(movie_list)}

    df['user_final'] = df['user_id'].map(user_map)
    df['movie_final'] = df['movie_id'].map(movie_map)

    df = df[['user_final', 'movie_final', 'rating']]

    X_train = df_to_numpy(df_train, user_map, movie_map)
    X_test_1 = df_to_numpy(df_test_1, user_map, movie_map)
    X_test_2 = df_to_numpy(df_test_2, user_map, movie_map)

    return (X_train, X_test_1, X_test_2), user_list, movie_list


def train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the given ratings dataframe into the following:
    test_2: The latest 10% of the data
    test_2: Of the remaining 90%, randomly sampled 10%
    train: Remaining 80%
    Assumes df has `timestamp` column
    """

    df.sort_values(by='timestamp')
    test_2_split = int(len(df) * 0.9)
    test_2 = df[test_2_split:]
    df = df[:test_2_split]
    df = df.sample(frac=1).reset_index(drop=True)
    test_1 = df[:len(test_2)]
    train = df[len(test_2):]

    return train, test_1, test_2


def df_to_numpy(df: pd.DataFrame) -> np.ndarray:
    """
    Converts dataframe with user_id, movie_id, ratings columns into np.ndarray ratings matrix where
    - row corresponds to user_id index
    - column corresponds to movie_id index
    - value corresponds to ratings
    Returns the ratings matrix
    """

    ratings_mtx = df.pivot(index='user_final', columns='movie_final', values='rating').to_numpy()

    return ratings_mtx
