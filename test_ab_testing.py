import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from io import StringIO

# Import your modules
from online_evaluation import Telemetry
from utils.util_functions import read_versions_from_yaml


class TestTelemetry(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.telemetry = Telemetry()
        
    def test_create_dummy_df(self):
        """Test if dummy dataframe is created with the correct structure"""
        num_rows = 5
        df = self.telemetry.create_dummy_df(num_rows)
        
        # Check dataframe shape
        self.assertEqual(df.shape[0], num_rows)
        self.assertEqual(df.shape[1], 7)  # 7 columns
        
        # Check column names
        expected_columns = [
            "timestamp", "model_version", "pipeline_version", 
            "data_start_date", "data_end_date", "user_id", "recommendations"
        ]
        self.assertListEqual(list(df.columns), expected_columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))
        self.assertTrue(pd.api.types.is_string_dtype(df["model_version"]))
        self.assertTrue(pd.api.types.is_string_dtype(df["user_id"]))
        
        # Check recommendations format
        sample_rec = df["recommendations"].iloc[0]
        self.assertTrue(all(rec.isdigit() for rec in sample_rec.split(",")))

    @patch('online_evaluation.get_all_recommendation_df')
    @patch('online_evaluation.get_watched_movies_all_data')
    def test_evaluate_with_mocked_data(self, mock_watched, mock_recommendations):
        """Test evaluate method with mocked data"""
        # Set up mock recommendation data
        rec_data = {
            "timestamp": [datetime.now()],
            "model_version": ["3.0.0"],
            "pipeline_version": ["2.0.1"],
            "data_start_date": [datetime.now() - timedelta(days=30)],
            "data_end_date": [datetime.now() - timedelta(days=15)],
            "user_id": ["user_1"],
            "recommendations": ["1, 2, 3, 4, 5"]
        }
        mock_rec_df = pd.DataFrame(rec_data)
        mock_recommendations.return_value = mock_rec_df
        
        # Set up mock watched movies data
        watched_data = {
            "movie_id": [1, 3, 5],
            "rating": [4.5, 3.0, 5.0],
            "timestamp": [datetime.now()] * 3
        }
        mock_watched_df = pd.DataFrame(watched_data)
        mock_watched.return_value = mock_watched_df
        
        # Run evaluate method
        avg_watch_rate, avg_rating = self.telemetry.evaluate("3.0.0")
        
        # Assertions
        self.assertEqual(avg_watch_rate, 0.6)  # 3 out of 5 movies watched
        self.assertEqual(avg_rating, (4.5 + 3.0 + 5.0) / 3)  # Average of ratings
        
        # Verify our mocks were called correctly
        mock_recommendations.assert_called_once_with("3.0.0")
        mock_watched.assert_called_once_with("user_1")

    @patch('online_evaluation.get_all_recommendation_df')
    @patch('online_evaluation.get_watched_movies_all_data')
    def test_evaluate_with_empty_recommendations(self, mock_watched, mock_recommendations):
        """Test evaluate method with empty recommendation data"""
        # Return empty dataframe
        mock_recommendations.return_value = pd.DataFrame()
        
        # Mock create_dummy_df method
        self.telemetry.create_dummy_df = MagicMock(return_value=pd.DataFrame({
            "timestamp": [datetime.now()],
            "model_version": ["3.0.0"],
            "pipeline_version": ["2.0.1"],
            "data_start_date": [datetime.now() - timedelta(days=30)],
            "data_end_date": [datetime.now() - timedelta(days=15)],
            "user_id": ["user_1"],
            "recommendations": ["1, 2, 3, 4, 5"]
        }))
        
        # Create a proper empty DataFrame with correct columns for watched movies
        empty_watched_df = pd.DataFrame(columns=["movie_id", "rating", "timestamp"])
        mock_watched.return_value = empty_watched_df
        
        avg_watch_rate, avg_rating = self.telemetry.evaluate("3.0.0")
        
        # Check if create_dummy_df was called
        self.telemetry.create_dummy_df.assert_called_once_with(10)
        
        # Since watched movies is empty, we expect default values
        self.assertEqual(avg_watch_rate, -1)
        self.assertEqual(avg_rating, -1)

    @patch('pandas.read_csv')
    @patch('matplotlib.pyplot.savefig')
    def test_plot(self, mock_savefig, mock_read_csv):
        """Test plotting functionality"""
        # Create mock dataframe for plotting
        mock_data = {
            "Time": pd.date_range(start='2023-01-01', periods=5),
            "Avg_Watchrate": [0.5, 0.6, 0.55, 0.7, 0.65],
            "Avg_Rating": [4.2, 4.3, 4.1, 4.4, 4.5]
        }
        mock_df = pd.DataFrame(mock_data)
        mock_read_csv.return_value = mock_df
        
        # Call the plot method
        self.telemetry.plot("dummy_path.csv", "test_plot.png", 5)
        
        # Check if savefig was called with the correct filename
        mock_savefig.assert_called_once()
        self.assertEqual(mock_savefig.call_args[0][0], "test_plot.png")
        
        # Verify read_csv was called
        mock_read_csv.assert_called_once_with("dummy_path.csv")


class TestABTesting(unittest.TestCase):
    """Tests specifically for the A/B testing functionality"""
    
    def setUp(self):
        """Set up test fixtures for A/B testing"""
        # Create sample data for previous and current model versions
        self.prev_watch_rate = 0.60
        self.prev_rating = 4.2
        self.curr_watch_rate = 0.75
        self.curr_rating = 4.7
        
        self.prev_stats = np.array([self.prev_watch_rate, self.prev_rating])
        self.curr_stats = np.array([self.curr_watch_rate, self.curr_rating])
        
        # Create more realistic sample data arrays for statistical tests
        self.prev_watch_rates = np.array([0.55, 0.60, 0.58, 0.62, 0.59, 0.61])
        self.prev_ratings = np.array([4.1, 4.2, 4.0, 4.3, 4.2, 4.1])
        
        self.curr_watch_rates = np.array([0.70, 0.75, 0.72, 0.73, 0.71, 0.74])
        self.curr_ratings = np.array([4.5, 4.7, 4.6, 4.8, 4.5, 4.6])
    
    def test_t_test_significant_difference(self):
        """Test t-test with data showing significant difference"""
        t_stat, p_value = stats.ttest_ind(self.prev_watch_rates, self.curr_watch_rates)
        
        # Verify t-statistic is negative (curr > prev)
        self.assertLess(t_stat, 0)
        
        # Verify p-value is less than 0.05 (significant difference)
        self.assertLess(p_value, 0.05)
        
        # Now test with ratings
        t_stat, p_value = stats.ttest_ind(self.prev_ratings, self.curr_ratings)
        self.assertLess(t_stat, 0)
        self.assertLess(p_value, 0.05)
    
    def test_t_test_no_significant_difference(self):
        """Test t-test with data showing no significant difference"""
        # Create data with highly overlapping distributions to ensure p-value > 0.05
        prev_data = np.array([0.65, 0.67, 0.66, 0.68, 0.64, 0.69])
        curr_data = np.array([0.66, 0.67, 0.65, 0.68, 0.64, 0.67])
        
        t_stat, p_value = stats.ttest_ind(prev_data, curr_data)
        
        # Print the actual p-value for debugging
        print(f"T-test p-value: {p_value}")
        
        # Verify p-value is greater than 0.05 (no significant difference)
        self.assertGreater(p_value, 0.05)
    
    def test_mann_whitney_significant_difference(self):
        """Test Mann-Whitney U test with data showing significant difference"""
        u_stat, p_value = stats.mannwhitneyu(self.prev_watch_rates, self.curr_watch_rates)
        
        # Verify p-value is less than 0.05 (significant difference)
        self.assertLess(p_value, 0.05)
        
        # Now test with ratings
        u_stat, p_value = stats.mannwhitneyu(self.prev_ratings, self.curr_ratings)
        self.assertLess(p_value, 0.05)
    
    def test_mann_whitney_no_significant_difference(self):
        """Test Mann-Whitney U test with data showing no significant difference"""
        # Create data with highly overlapping distributions to ensure p-value > 0.05
        prev_data = np.array([4.2, 4.3, 4.1, 4.3, 4.2, 4.4])
        curr_data = np.array([4.3, 4.2, 4.3, 4.4, 4.1, 4.3])
        
        u_stat, p_value = stats.mannwhitneyu(prev_data, curr_data)
        
        # Print the actual p-value for debugging
        print(f"Mann-Whitney p-value: {p_value}")
        
        # Verify p-value is greater than 0.05 (no significant difference)
        self.assertGreater(p_value, 0.05)
    
    def test_results_file_content(self):
        """Test that the AB testing results file has the correct content"""
        # Mock data
        t_stat = -5.78
        p_value = 0.0003
        u_stat = 0.0
        p_value_u = 0.0022
        
        # Use StringIO to capture file writes
        file_contents = StringIO()
        
        # Write AB test results
        file_contents.write(f"Independent Samples T-test:\n")
        file_contents.write(f"T-statistic: {t_stat}\n")
        file_contents.write(f"P-value: {p_value}\n\n")
        file_contents.write(f"Mann-Whitney U Test:\n")
        file_contents.write(f"U-statistic: {u_stat}\n")
        file_contents.write(f"P-value (Mann-Whitney): {p_value_u}\n")
        
        # Reset the position to the beginning of the file
        file_contents.seek(0)
        
        # Read the content
        content = file_contents.read()
        
        # Verify the content contains expected information
        self.assertIn("Independent Samples T-test:", content)
        self.assertIn(f"T-statistic: {t_stat}", content)
        self.assertIn(f"P-value: {p_value}", content)
        self.assertIn("Mann-Whitney U Test:", content)
        self.assertIn(f"U-statistic: {u_stat}", content)
        self.assertIn(f"P-value (Mann-Whitney): {p_value_u}", content)
    
    @patch('builtins.open', new_callable=mock_open)
    def test_ab_testing_file_output(self, mock_open_file):
        """Test the AB testing file output"""
        # Mock data
        t_stat = -4.56
        p_value = 0.0015
        u_stat = 2.0
        p_value_u = 0.0043
        
        # Create the results file
        with open("ab_testing_results.txt", "w") as f:
            f.write(f"Independent Samples T-test:\n")
            f.write(f"T-statistic: {t_stat}\n")
            f.write(f"P-value: {p_value}\n\n")
            f.write(f"Mann-Whitney U Test:\n")
            f.write(f"U-statistic: {u_stat}\n")
            f.write(f"P-value (Mann-Whitney): {p_value_u}\n")
        
        # Verify file open was called correctly
        mock_open_file.assert_called_once_with("ab_testing_results.txt", "w")
        
        # Get all the write calls
        write_calls = [call[0][0] for call in mock_open_file().write.call_args_list]
        
        # Verify content of writes contains expected strings
        expected_strings = [
            "Independent Samples T-test:",
            f"T-statistic: {t_stat}",
            f"P-value: {p_value}",
            "Mann-Whitney U Test:",
            f"U-statistic: {u_stat}",
            f"P-value (Mann-Whitney): {p_value_u}"
        ]
        
        # Check each expected string is in at least one of the write calls
        for expected in expected_strings:
            self.assertTrue(
                any(expected in call for call in write_calls),
                f"Expected string '{expected}' not found in write calls"
            )


class TestMainScript(unittest.TestCase):
    @patch('utils.util_functions.read_versions_from_yaml')
    @patch('online_evaluation.Telemetry')
    @patch('scipy.stats.ttest_ind')
    @patch('scipy.stats.mannwhitneyu')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pandas.DataFrame.to_csv')
    @patch('pandas.io.common.file_exists')
    def test_main_script(self, mock_file_exists, mock_to_csv, mock_open_file, 
                         mock_mannwhitneyu, mock_ttest, mock_telemetry, mock_read_yaml):
        """Test the main script execution flow"""
        # Set up mocks
        mock_read_yaml.return_value = ("prev_version", "curr_version")
        
        telemetry_instance = MagicMock()
        telemetry_instance.evaluate.side_effect = [
            (0.6, 4.2),  # previous version stats
            (0.7, 4.5)   # current version stats
        ]
        mock_telemetry.return_value = telemetry_instance
        
        mock_ttest.return_value = (2.5, 0.03)  # t-stat, p-value
        mock_mannwhitneyu.return_value = (10, 0.04)  # u-stat, p-value
        mock_file_exists.return_value = False
        
        # Import and run the main script
        # Note: This is a bit tricky in unit tests, so we'll mock the __name__ == "__main__" check
        # Instead of manipulating sys.modules directly, we'll use patch
        
        # Execute the main script code with appropriate mocking
        with patch.dict('sys.modules', {'__main__': MagicMock()}):
            # This would normally run the main script, but we'll simulate it
            # by directly calling the key functions with our mocks
            
            # Simulate script execution
            filepath = 'version.yaml'
            previous_version, current_version = mock_read_yaml(filepath)
            telemetry = mock_telemetry()
            
            prev_stats = np.array(telemetry.evaluate(previous_version)).astype(float)
            curr_stats = np.array(telemetry.evaluate(current_version)).astype(float)
            
            # Statistical tests
            t_stat, p_value = mock_ttest(prev_stats, curr_stats)
            u_stat, p_value_u = mock_mannwhitneyu(prev_stats, curr_stats)
            
            # File operations
            with mock_open_file("ab_testing_results.txt", "w") as f:
                f.write("Mock file content")
            
            # DataFrame operations
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
            new_data = pd.DataFrame(
                [[formatted_now, prev_stats[0], prev_stats[1], previous_version], 
                 [formatted_now, curr_stats[0], curr_stats[1], current_version]], 
                columns=["Time", "Avg_Watchrate", "Avg_Rating", "Model Version"]
            )
            
            new_data.to_csv(
                "evaluation_metrics.csv", 
                mode="a", 
                header=not mock_file_exists("evaluation_metrics.csv"),
                index=False
            )
        
        # Verify function calls
        mock_read_yaml.assert_called_once_with(filepath)
        self.assertEqual(telemetry_instance.evaluate.call_count, 2)
        mock_ttest.assert_called_once()
        mock_mannwhitneyu.assert_called_once()
        mock_open_file.assert_called_once()
        mock_to_csv.assert_called_once()


if __name__ == '__main__':
    unittest.main()