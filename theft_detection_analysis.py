#!/usr/bin/env python3
# theft_detection_analysis.py
"""
Simplified Model Theft Detection for SVD-based Recommendation Models
Designed for reliability and speed with large datasets
"""

import sys
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_theft_detector.log')
    ]
)
logger = logging.getLogger('model_theft_detector')

# Import your database functions
from utils.sql import get_all_recommendation_df, get_watched_movies_all_data


def detect_model_theft():
    """
    Simple function to detect model theft attempts based on user behavior patterns.
    """
    start_time = time.time()
    logger.info("Starting model theft detection")
    
    try:
        # 1. Get recommendation data
        rc_df = get_all_recommendation_df(model_version=None)
        logger.info(f"Retrieved {len(rc_df)} recommendation records")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(rc_df['timestamp']):
            rc_df['timestamp'] = pd.to_datetime(rc_df['timestamp'])
        
        # 2. Count recommendations per user
        user_counts = rc_df['user_id'].value_counts()
        logger.info(f"Analyzed request patterns for {len(user_counts)} unique users")
        
        # 3. Find high-volume users (potential suspicious users)
        HIGH_REQUEST_THRESHOLD = 24
        high_volume_users = user_counts[user_counts > HIGH_REQUEST_THRESHOLD].index.tolist()
        logger.info(f"Found {len(high_volume_users)} users with high request volume")
        
        # If no high-volume users, we're done
        if not high_volume_users:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Detection complete in {execution_time:.2f} seconds. Found 0 suspicious users")
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": execution_time,
                "detection_results": {
                    "suspicious_user_count": 0,
                    "suspicious_users": []
                }
            }
            
            return report
        
        # 4. For high-volume users, check for regular patterns and low engagement
        suspicious_users = []
        
        for user_id in high_volume_users:
            # Get user's requests
            user_df = rc_df[rc_df['user_id'] == user_id].sort_values('timestamp')
            
            # Define flags to track
            suspicious_flags = []
            suspicious_flags.append("high_request_volume")
            
            # Check request regularity
            if len(user_df) > 5:
                intervals = []
                for i in range(len(user_df) - 1):
                    delta = (user_df.iloc[i+1]['timestamp'] - user_df.iloc[i]['timestamp']).total_seconds()
                    intervals.append(delta)
                
                if intervals:
                    mean_interval = np.mean(intervals)
                    if mean_interval > 0:
                        std_dev = np.std(intervals)
                        # Coefficient of variation
                        regularity_score = std_dev / mean_interval
                        if regularity_score < 0.2:  # Very regular timing
                            suspicious_flags.append("regular_request_pattern")
            
            # Only check watch rate if already suspicious
            if True:
                # Get user's watched movies
                watched_df = get_watched_movies_all_data(user_id)
                watched_movie_ids = set()
                
                if not watched_df.empty:
                    watched_movie_ids = set(watched_df['movie_id'].astype(str).tolist())
                
                # Get recommended movies
                recommended_movies = []
                for _, row in user_df.iterrows():
                    print(row)
                    movie_ids = row['recommendations'].split(',')
                    recommended_movies.extend([m.strip() for m in movie_ids])
                
                if recommended_movies:
                    # Calculate watch rate
                    watched_count = sum(1 for m in recommended_movies if m in watched_movie_ids)
                    watch_rate = watched_count / len(recommended_movies) if recommended_movies else 0
                    print("WATCH RATE ", watch_rate)
                    # Check for low engagement
                    if watch_rate < 0.1:
                        suspicious_flags.append("low_engagement")
                        
                    # Add to suspicious list if multiple flags
                    if len(suspicious_flags) >= 2:
                        user_info = {
                            "user_id": user_id,
                            "flags": suspicious_flags,
                            "request_count": len(user_df),
                            "watch_rate": watch_rate
                        }
                        suspicious_users.append(user_info)
        
        # 5. Create detection report
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"Detection complete in {execution_time:.2f} seconds. Found {len(suspicious_users)} suspicious users")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": execution_time,
            "detection_results": {
                "suspicious_user_count": len(suspicious_users),
                "suspicious_users": suspicious_users
            }
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Error in detection: {e}")
        # Return basic error report
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


if __name__ == "__main__":
    try:
        # Run detection
        detection_results = detect_model_theft()
        
        # Save results to file
        output_file = "model_theft_report.json"
        with open(output_file, 'w') as f:
            json.dump(detection_results, f, indent=2)
        
        # Log results summary
        suspicious_count = detection_results.get('detection_results', {}).get('suspicious_user_count', 0)
        execution_time = detection_results.get('execution_time_seconds', 0)
        print(f"Detection complete in {execution_time:.2f} seconds.")
        print(f"Found {suspicious_count} suspicious users.")
        print(f"Results saved to {output_file}")
            
    except Exception as e:
        logger.error(f"Error during model theft detection: {e}")
        print(f"ERROR: Model theft detection failed: {e}", file=sys.stderr)
        sys.exit(1)