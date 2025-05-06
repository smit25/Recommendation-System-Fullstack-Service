import scipy.stats as stats
import numpy as np
from datetime import datetime
import pandas as pd

from online_evaluation import Telemetry
from utils.util_functions import read_versions_from_yaml

if __name__ == "__main__":
    filepath = 'version.yaml'

    previous_version, current_version = read_versions_from_yaml(filepath)
    telemetry = Telemetry()

    # avg_watch_rate_prev, avg_rating_prev = telemetry.evaluate(previous_version)
    # avg_watch_rate_curr, avg_rating_curr = telemetry.evaluate(current_version)

    telemetry_previous_v = "svd_model_" + previous_version.replace(".", "_")
    telemetry_current_v = "svd_model_" + current_version.replace(".", "_")

    curr_stats = np.array(telemetry.evaluate(telemetry_current_v)).astype(float)
    prev_stats = np.array(telemetry.evaluate(telemetry_previous_v)).astype(float)

    # # Independent samples t-test
    # t_stat, p_value = stats.ttest_ind(prev_stats, curr_stats)
    # print(f"T-statistic: {t_stat}, P-value: {p_value}")

    # Mann-Whitney U test
    u_stat, p_value_u = stats.mannwhitneyu(prev_stats, curr_stats)
    print(f"U-statistic: {u_stat}, P-value (Mann-Whitney): {p_value_u}")

    with open("ab_testing_results.txt", "w") as f:
        # f.write(f"Independent Samples T-test:\n")
        # f.write(f"T-statistic: {t_stat:.2f}\n")
        # f.write(f"P-value: {p_value:.2f}\n\n")
        f.write(f"Previous Version: {previous_version}\n")
        f.write(f"Current Version: {current_version}\n\n")

        f.write(f"Mann-Whitney U Test:\n")
        f.write(f"U-statistic: {u_stat:.2f}\n")
        f.write(f"P-value (Mann-Whitney): {p_value_u:.2f}\n")

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([[formatted_now, prev_stats[0], prev_stats[1], previous_version], [formatted_now, curr_stats[0], curr_stats[1], current_version]], 
                        columns=["Time", "Avg_Watchrate", "Avg_Rating", "Model Version"])

    new_data.to_csv(
        "evaluation_metrics.csv", 
        mode="a", 
        header=not pd.io.common.file_exists("evaluation_metrics.csv"),
        index=False
    )






