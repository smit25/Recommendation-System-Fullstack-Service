import pandas as pd
from sqlalchemy import create_engine, text
import re
from utils.sql import get_engine,flush_table

def clean_timestamp(x):
    x=re.sub(r'[^0-9T:-]', '', x)
    return x[:16]

def clean_user_id(x):
    x=re.sub(r'[^0-9]', '', x)
    return x

def clean_ratings(df):
    df['movie_id'] = df['content'].str.split('/').str[-1].str[:-2]
    df['rating'] = pd.to_numeric(df['content'].str[-1], errors='coerce')
    df=df.dropna(subset=['rating'])
    df['rating'] = df['content'].str[-1].astype(int)
    return df[['timestamp', 'user_id', 'movie_id', 'rating']]

def clean_watches(df):
    df['movie_id'] = df['content'].str.split('/').str[-2]
    df['minute'] = df['content'].str.extract(r'GET\s+/data/m/[^/]+/(\d+).*\.mpg')
    df=df.dropna(subset=['minute'])
    df['minute']=df['minute'].astype(int)
    return df[['timestamp', 'user_id', 'movie_id', 'minute']].sort_values(['user_id', 'movie_id', 'minute'], ascending=False).drop_duplicates(subset=['user_id', 'movie_id'], keep='first')

# def clean_recommend(df):
#     df['recommendations'] = df['content'].str.extract(r'.*result: (.*?), \d+ ms')
#     return df[['timestamp', 'user_id', 'recommendations']].dropna(subset='recommendations')

def main():
    with open('kafka_stream_ingest.txt', 'r') as f:
        rows = (line.rstrip().split(',', 2) for line in f)
        df = pd.DataFrame(rows, columns=['timestamp', 'user_id', 'content'])
    df['timestamp'] = df['timestamp'].apply(clean_timestamp)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['user_id'] = df['user_id'].apply(clean_user_id)
    df = df.dropna(subset=['timestamp'])

    df_ratings = df[df['content'].str[5:9] == 'rate']
    df_watches = df[df['content'].str[5:9] == 'data']
    # df_recommend = df[df['content'].str.startswith('recommend')]

    df_ratings = clean_ratings(df_ratings)
    df_watches = clean_watches(df_watches)
    # df_recommend = clean_recommend(df_recommend)
    print("Processed")

    engine = get_engine()
    df_ratings.to_sql('user_ratings', engine, if_exists='append', index=False)
    df_watches.to_sql('watch', engine, if_exists='append', index=False)

    print("Saved to DB")

    # with engine.begin() as con:
    #     flush_table('watch','48 hours',engine)
    #     flush_table('recommend','48 hours',engine)

    print("Flushed old Entries from DB")


if __name__ == "__main__":
    main()
