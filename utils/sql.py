import configparser
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text

def get_engine():
    config = configparser.ConfigParser()
    config.read('config.ini')

    username = config.get('database', 'username')
    password = config.get('database', 'password')
    host = config.get('database', 'host')
    port = config.get('database', 'port')
    schema = config.get('database', 'schema')
    engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{schema}')
    return engine


def flush_table(table_name: str, interval='48 hours', engine=None) -> None:
    if engine is None:
        engine = get_engine()

    query = f"DELETE FROM {table_name} WHERE timestamp < NOW() - INTERVAL '{interval}'"
    
    with engine.begin() as con:
        con.execute(text(query))


def get_watched_movies(user_id: str, engine=None) -> list[str]:
    if engine == None:
        engine = get_engine()

    query = f"SELECT DISTINCT movie_id FROM user_ratings WHERE user_id = '{user_id}'"
    
    with engine.begin() as con:
        result = con.execute(text(query))
    
    return [r[0] for r in result]


def get_ratings_df(engine=None) -> pd.DataFrame:
    if engine == None:
        engine = get_engine()

    with engine.begin() as con:
        df = pd.read_sql_table('user_ratings', con)
    
    return df

def get_all_recommendation_df(model_version: str, engine= None) -> pd.DataFrame:
    if engine == None:
        engine = get_engine()

    # model_version = model_version.replace(".", "_")
    # query = f"SELECT DISTINCT * FROM recommend WHERE model_version = '{model_version}' LIMIT 10000"
    query = f"SELECT DISTINCT * FROM recommend WHERE model_version = '{model_version}' ORDER BY timestamp DESC LIMIT 1000"

    with engine.begin() as con:
        df = pd.read_sql_query(query, con)
    
    return df

def get_watched_movies_all_data(user_id: str, engine = None) -> pd.DataFrame:
    if engine == None:
        engine = get_engine()
    
    query = f"SELECT DISTINCT movie_id, rating, timestamp FROM user_ratings WHERE user_id = '{user_id}'"
    with engine.begin() as con:
        df = pd.read_sql_query(query, con)
    
    return df


def log_prediction(model_version: str, pipeline_version: str, data_start_date, data_end_date, user_id: str, recommendations: list[str], engine=None) -> None:
    if engine is None:
        engine = get_engine()

    timestamp = datetime.now()
    content = ', '.join(recommendations)

    query = text("INSERT INTO recommend VALUES (:timestamp, :model_version, :pipeline_version, :data_start_date, :data_end_date, :user_id, :content)")
    params = {
        "timestamp": str(timestamp),
        "model_version": model_version,
        "pipeline_version": pipeline_version,
        "data_start_date": data_start_date,
        "data_end_date": data_end_date,
        "user_id": user_id,
        "content": content
    }

    with engine.begin() as con:
        con.execute(query, params)
