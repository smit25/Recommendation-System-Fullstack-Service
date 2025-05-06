import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from utils.sql import get_engine

engine = get_engine()

df = pd.read_sql(
    f"SELECT timestamp, user_id, movie_id FROM watch",
    engine,
    parse_dates=['timestamp']
)

top5 = df['movie_id'].value_counts().nlargest(5).index.tolist()

df_top5 = df[df['movie_id'].isin(top5)].copy()
df_top5.set_index('timestamp', inplace=True)

daily_counts = (
    df_top5
    .groupby([pd.Grouper(freq='D'), 'movie_id'])
    .size()
    .unstack(fill_value=0)
    .loc[:, top5]  
)

plt.figure(figsize=(10, 6))
for movie_id in top5:
    plt.plot(
        daily_counts.index,
        daily_counts[movie_id],
        marker='o',
        label=str(movie_id)
    )

plt.title('Daily Watch Count of Top 5 Movies')
plt.xlabel('Date')
plt.ylabel('Watch Count')
plt.legend(title='Movie ID')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
