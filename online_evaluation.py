import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from utils.sql import get_all_recommendation_df, get_watched_movies_all_data


class Telemetry:
    def __init__(self, _sample_size = 50):
        self.sample_size = _sample_size
    
    def create_dummy_df(self, num_rows = 10):
        timestamps = [datetime.now() - timedelta(days=i) for i in range(num_rows)]
        model_versions = [f"3.0.0" for i in range(num_rows)]
        pipeline_versions = [f"2.0.1" for i in range(num_rows)]
        data_start_dates = [datetime.now() - timedelta(days=30 + i) for i in range(num_rows)]
        data_end_dates = [datetime.now() - timedelta(days=15 + i) for i in range(num_rows)]
        user_ids = [f"user_{i}" for i in range(num_rows)]
        recommendations = [(",".join([str(j + 1) for j in range(20)])) for i in range(num_rows)]

        data = {
            "timestamp": timestamps,
            "model_version": model_versions,
            "pipeline_version": pipeline_versions,
            "data_start_date": data_start_dates,
            "data_end_date": data_end_dates,
            "user_id": user_ids,
            "recommendations": recommendations,
        }

        df = pd.DataFrame(data)
        return df


    def evaluate(self, model_version):
        """
        Evaluates if the movies recommended by the recommdation system are actually watched by the user.
        Metric 1: Average Watch Rate
        Metric 2: "Average Rating" of the recommended movies that were watched by the user.
        """
        unique_users = {}
        rc_df = get_all_recommendation_df(model_version)
        print(len(rc_df))
        print(model_version)

        print(rc_df.head())

        if rc_df.empty:
            rc_df = self.create_dummy_df(10)

        for _, item in rc_df.iterrows():
            _user_id = item["user_id"]
            unique_users[_user_id] = item
            # if len(unique_users) >= self.sample_size:
            #     break
        
        watch_rate = []
        watched_ratings = []
        print("Unique Users: ", len(unique_users))
    
        for _user_id, item in unique_users.items():
            rec_movies = item["recommendations"].split(",")
            rec_movies = [movie.strip() for movie in rec_movies]
            
            df_by_user = get_watched_movies_all_data(_user_id)
            print("user_id: ", _user_id)
            if df_by_user.empty:
                print("No data for user_id: ", _user_id)
                continue
            watched_movies = df_by_user["movie_id"].astype(str).tolist()
            movie_to_rating = df_by_user.set_index("movie_id")["rating"].to_dict()
            movie_to_rating = {str(k): v for k, v in movie_to_rating.items()}
            
            rec_movies_len = len(rec_movies)
            watched_ratings_per_user = []
            watched_movies_counter = 0
            
            if len(watched_movies) == 0 or rec_movies_len == 0:
                continue
            
            for movie in rec_movies:
                if movie in watched_movies:
                    watched_movies_counter += 1
                    rating = movie_to_rating.get(movie, -1)

                    if not pd.isna(rating) and rating != -1:
                        watched_ratings_per_user.append(rating)

            watch_rate.append(watched_movies_counter / rec_movies_len)

            if len(watched_ratings_per_user) > 0:
                watched_ratings.append(sum(watched_ratings_per_user) / len(watched_ratings_per_user)) 

            
        avg_watch_rate = sum(watch_rate) / len(watch_rate) if watch_rate else -1
        avg_rating = sum(watched_ratings) / len(watched_ratings) if watched_ratings else -1
        
        return avg_watch_rate, avg_rating
    

    def plot(self, evaluation_metrics_path, evaluation_image = "telemetry.png", metrics_len = 30):
        df = pd.read_csv(evaluation_metrics_path)
        df = df.tail(metrics_len)

        df["Time"] = pd.to_datetime(df["Time"])

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

        axes[0].plot(df["Time"], df["Avg_Watchrate"], label="Avg Watch Rate", marker="o", linestyle="-", color="blue")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Avg Watch Rate")
        axes[0].set_title("Average Watch Rate Over Time")
        axes[0].legend()
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(True)

        axes[1].plot(df["Time"], df["Avg_Rating"], label="Avg Rating", marker="s", linestyle="--", color="red")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Avg Rating")
        axes[1].set_title("Average Rating Over Time")
        axes[1].legend()
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True)

        plt.tight_layout()

        plt.savefig(evaluation_image, dpi=300, bbox_inches="tight")

        # plt.show()



# if __name__ == "__main__":
#     # if len(sys.argv) != 3:
#     #     print("Usage: python telemetry.py <recommendation_path> <data_path>")
#     #     sys.exit(1)

#     # recommendation_path = sys.argv[1]
#     # data_path = sys.argv[2]
    
#     evaluation_metrics_path = "evaluation_metrics.csv"
#     telemetry = Telemetry()
#     avg_watch_rate, avg_rating = telemetry.evaluate()
#     print(avg_watch_rate, avg_rating)

#     if avg_rating == -1:
#         avg_rating = ""
#     if avg_watch_rate == -1:
#         avg_watch_rate = ""

#     now = datetime.now()
#     formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
#     new_data = pd.DataFrame([[formatted_now, avg_watch_rate, avg_rating]], 
#                         columns=["Time", "Avg_Watchrate", "Avg_Rating"])

#     new_data.to_csv(
#         evaluation_metrics_path, 
#         mode="a", 
#         header=not pd.io.common.file_exists(evaluation_metrics_path),
#         index=False
#     )
#     telemetry.plot(evaluation_metrics_path)
        




