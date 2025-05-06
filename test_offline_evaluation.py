import unittest
import pandas as pd

from offline_evaluation import OfflineEvaluation
from model.matrix_factorization import MatrixFactorization

class TestOfflineEvaluation(unittest.TestCase):
    def setUp(self):
        self.oe = OfflineEvaluation()

        # Create dummy DataFrames for train and tests.
        self.oe.df_train  = pd.DataFrame({'timestamp': [0,0,0,0,0],
                                          'user_id': [1,1,2,2,3],
                                          'movie_id': [1,2,2,3,1],
                                          'rating': [1,2,3,4,5]})
        self.oe.df_test_1 = pd.DataFrame({'timestamp': [0,0,0],
                                          'user_id': [1,2,3],
                                          'movie_id': [3,1,2],
                                          'rating': [1,2,3]})
        self.oe.df_test_2 = pd.DataFrame({'timestamp': [0,0,0],
                                          'user_id': [1,2,3],
                                          'movie_id': [3,1,3],
                                          'rating': [1,2,3]})

        self.oe.model = MatrixFactorization(version='test', pipeline_version='test')
        self.oe.model.train(self.oe.df_train)

    def test_mses(self):
        mse_1, mse_2 = self.oe.mses()
        self.assertAlmostEqual(mse_1, 1.2910, places=4)
        self.assertAlmostEqual(mse_2, 1.2910, places=4)

    def test_data_distributions(self):
        (res_mean1, res_std1), (res_mean2, res_std2), res_t_stat, res_p_val = self.oe.data_distributions()

        self.assertEqual(res_mean1, 2.0)
        self.assertEqual(res_mean2, 2.0)
        self.assertAlmostEqual(res_std1, 0.8165, places=4)
        self.assertAlmostEqual(res_std2, 0.8165, places=4)
        self.assertEqual(res_t_stat, 0)
        self.assertEqual(res_p_val, 1)

if __name__ == '__main__':
    unittest.main()
