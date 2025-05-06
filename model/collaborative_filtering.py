from utils.clean_data import clean_data
import numpy as np


class CollaborativeFilteringRecommender:
    def __init__(self):
        self.ratings_mtx, self.user_list, self.movie_list = clean_data()

    def train(X: np.ndarray):
        pass

    def cosine_similarity(u: np.array, v: np.array) -> float:
        """
        Helper function for measuring cosine similarity between two vectors.
        Returns 0.0 if there is no similarity (no non-NaN intersections)
        """
        mask = ~np.isnan(u) & ~np.isnan(v)
        if np.sum(mask) == 0:
            return 0.0
        u_ = u[mask]
        v_ = v[mask]
        u_norm = np.linalg.norm(u_)
        v_norm = np.linalg.norm(v_)
        if u_norm == 0 or v_norm == 0:
            return 0.0
        return np.dot(u_, v_) / (u_norm * v_norm)

    def recommend_movies(self, user_id: int, n=3) -> list[str]:
        """
        Returns the top n recommended movies for a user with given user_id
        """
        try:
            user_idx = self.user_list.index(user_id)
        except ValueError:
            return ["Popular Movie 1", "Popular Movie 2", "Popular Movie 3"]
        
        target_ratings = self.ratings_mtx[user_idx]
        num_users, num_movies = self.ratings_mtx.shape
        
        # Compute similarity between the target user and all other users.
        similarities = np.zeros(num_users)
        for i in range(num_users):
            if i == user_idx:
                similarities[i] = 0.0
            else:
                similarities[i] = CollaborativeFilteringRecommender.cosine_similarity(target_ratings, self.ratings_mtx[i])
        
        # For each movie not rated by the target user, predict a rating.
        predicted_ratings = np.full(num_movies, np.nan)
        for m in range(num_movies):
            if not np.isnan(target_ratings[m]):
                continue
            
            rated_mask = ~np.isnan(self.ratings_mtx[:, m])
            if np.sum(rated_mask) == 0:
                continue
            
            sim_scores = similarities[rated_mask]
            ratings = self.ratings_mtx[rated_mask, m]
            
            # If the sum of absolute similarities is zero, skip prediction.
            if np.sum(np.abs(sim_scores)) == 0:
                continue
            
            predicted_rating = np.dot(sim_scores, ratings) / np.sum(np.abs(sim_scores))
            predicted_ratings[m] = predicted_rating

        # Get indices of movies with a predicted rating and sort them descending.
        valid_indices = np.where(~np.isnan(predicted_ratings))[0]
        
        sorted_indices = valid_indices[np.argsort(predicted_ratings[valid_indices])[::-1]]
        top_n_indices = sorted_indices[:min(n, len(valid_indices))]
        
        # Map indices to movie names.
        recommendations = [self.movie_list[i] for i in top_n_indices]
        for i in range(min(n - len(valid_indices), n)):
            recommendations.append(f'Popular Movie {i + 1}')
        return recommendations
    
    def evaluate(self, subset=0.01) -> int:
        """
        Returns the MSE of the collaborative filtering algorithm.
        The subset argument determines a randomly selected subset of values that will be tested
        """

        # Randomly uniformly select non-null values in ratings matrix
        non_null_idx = np.argwhere(~np.isnan(self.ratings_mtx))
        sample_size = max(1, int(subset * len(non_null_idx)))
        selected_idx = non_null_idx[np.random.choice(len(non_null_idx), size=sample_size, replace=False)]

        num_users, _ = self.ratings_mtx.shape
        errors = []
        count = 0
        for i, j in selected_idx:
            count += 1
            print(f'Evaluating point {count} / {sample_size}')
            actual = self.ratings_mtx[i, j]
            if not np.isnan(actual):
                self.ratings_mtx[i, j] = np.nan

                # Compute similarity between the target user and all other users.
                target_ratings = self.ratings_mtx[i]
                similarities = np.zeros(num_users)
                for k in range(num_users):
                    if k == i:
                        similarities[k] = 0.0
                    else:
                        similarities[k] = CollaborativeFilteringRecommender.cosine_similarity(target_ratings, self.ratings_mtx[k])
                
                # For each movie not rated by the target user, predict a rating.
                rated_mask = ~np.isnan(self.ratings_mtx[:, j])
                if np.sum(rated_mask) == 0:
                    continue
                    
                sim_scores = similarities[rated_mask]
                ratings = self.ratings_mtx[rated_mask, j]
                    
                # If the sum of absolute similarities is zero, skip prediction.
                if np.sum(np.abs(sim_scores)) == 0:
                    continue
                    
                predicted_rating = np.dot(sim_scores, ratings) / np.sum(np.abs(sim_scores))
                if not np.isnan(predicted_rating):
                    errors.append((predicted_rating - actual) ** 2)

                self.ratings_mtx[i, j] = actual
        
        return np.mean(errors) if errors else np.nan
