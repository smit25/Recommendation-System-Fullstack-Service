import numpy as np
import psutil
import os
import timeit
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.collaborative_filtering import CollaborativeFilteringRecommender


def memory_usage():
    """
    Returns memory usage of program in MB
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def training_time(cfr: CollaborativeFilteringRecommender):
    return timeit.timeit(cfr.train(np.ndarray([[0]])))


def inference_time(cfr: CollaborativeFilteringRecommender, n=10):
    """
    Returns average inference time across n runs
    """
    # helper function for timeit
    def prediction_call():
        id = np.random.choice(cfr.user_list, size=1)
        cfr.recommend_movies(id)
        return
    
    total_time = timeit.timeit(prediction_call, number=n)
    return total_time / n


cfr = CollaborativeFilteringRecommender()
initial_memory = memory_usage()
print(f'Memory usage after initialization: {initial_memory}MB')
training_time = training_time(cfr)
print(f'Training time: {training_time}')
training_memory = memory_usage()
print(f'Memory usage after training: {training_memory}MB')
avg_inference_time = inference_time(cfr)
print(f'Average inference time: {avg_inference_time}')
mse = cfr.evaluate()
print(f'MSE: {mse}')
final_memory = memory_usage()
print(f'Memory usage after inference: {final_memory}MB')
