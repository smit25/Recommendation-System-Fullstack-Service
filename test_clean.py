import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
from sqlalchemy import create_engine
from io import StringIO

from clean_data import (
    clean_timestamp,
    clean_user_id,
    clean_ratings,
    clean_watches,
    clean_recommend,
    main 
)

class TestDataCleaning(unittest.TestCase):

    def test_clean_timestamp(self):
        self.assertEqual(clean_timestamp('2025-03-01T12:45:30'), '2025-03-01T12:45')
        self.assertEqual(clean_timestamp('2025-03-01T12:45:30.123456'), '2025-03-01T12:45')
        self.assertEqual(clean_timestamp('invalid_timestamp_string'), '')

    def test_clean_user_id(self):
        self.assertEqual(clean_user_id('user1234'), '1234')
        self.assertEqual(clean_user_id('user#@1234'), '1234')
        self.assertEqual(clean_user_id('1234abc'), '1234')

    def test_clean_ratings(self):
        df = pd.DataFrame({
            'timestamp': ['2025-03-01T12:45', '2025-03-02T13:45'],
            'user_id': ['1234', '5678'],
            'content': ['/rate/life+is+beautiful+1997=5', '/rate/the+godfather+1972=4']
        })

        result_df = clean_ratings(df)
        self.assertEqual(result_df.shape[0], 2)
        self.assertTrue('movie_id' in result_df.columns)
        self.assertTrue('rating' in result_df.columns)

        self.assertEqual(result_df['movie_id'].iloc[0], 'life+is+beautiful+1997')
        self.assertEqual(result_df['rating'].iloc[0], 5)

    def test_clean_watches(self):
        df = pd.DataFrame({
            'timestamp': ['2025-03-01T12:45', '2025-03-02T13:45'],
            'user_id': ['1234', '5678'],
            'content': ['GET /data/m/the+corporation+2003/128.mpg', 'GET /data/m/oldboy+2003/90.mpg']
        })
        result_df = clean_watches(df)
        self.assertEqual(result_df.shape[0], 2)
        self.assertTrue('movie_id' in result_df.columns)
        self.assertTrue('minute' in result_df.columns)
        self.assertEqual(result_df['movie_id'].iloc[0], 'oldboy+2003')
        self.assertEqual(result_df['minute'].iloc[0], 90)

    def test_clean_recommend(self):
        df = pd.DataFrame({
            'timestamp': ['2025-03-01T12:45', '2025-03-02T13:45'],
            'user_id': ['1234', '5678'],
            'content': ['recommendation request 17645, result: the+shawshank+redemption+1994, life+is+beautiful+1997, 41 ms', 'recommendation request 17646, result: the+godfather+1972, 41 ms']
        })
        result_df = clean_recommend(df)
        self.assertEqual(result_df.shape[0], 2)
        self.assertTrue('recommendations' in result_df.columns)

        self.assertEqual(result_df['recommendations'].iloc[0], 'the+shawshank+redemption+1994, life+is+beautiful+1997')

    @patch('pandas.DataFrame.to_sql')
    def test_save_to_db(self, mock_to_sql):
        df = pd.DataFrame({
            'timestamp': ['2025-03-01T12:45'],
            'user_id': ['1234'],
            'content': ['/rate/life+is+beautiful+1997=5']
        })
        engine = MagicMock()
        df_ratings = clean_ratings(df)
        
        df_ratings.to_sql('user_ratings', engine, if_exists='append', index=False)
        self.assertTrue(mock_to_sql.called)

    @patch("builtins.open", new_callable=mock_open, read_data="2023-04-05T12:34,123,rate/movie/678/4\n")
    @patch("sqlalchemy.create_engine")
    def test_main(self, mock_engine, mock_file):
        main()

   
if __name__ == '__main__':
    unittest.main()

