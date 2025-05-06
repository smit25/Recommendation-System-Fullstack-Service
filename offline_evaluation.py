from utils.clean_data import clean_data
from model.matrix_factorization import MatrixFactorization
import numpy as np
from scipy import stats


class OfflineEvaluation():
    def __init__(self, version=None, pipeline_version=None):
        (df_train, df_test_1, df_test_2), _, _ = clean_data()
        self.df_train = df_train
        self.df_test_1 = df_test_1
        self.df_test_2 = df_test_2
        if version:
            self.model = MatrixFactorization(version, pipeline_version)
        else:
            model = MatrixFactorization(version='test', pipeline_version='test')
            model.train(df_train)
            self.model = model

    def mses(self) -> tuple[float, float]:
        """
        Returns a tuple of floats:
        (mse_1, mse_2)
        mse_1 is the mse of test set 1 (same timeframe as train)
        mse_2 is the mse of test set 2 (latest data)
        """
        mse_1 = self.model.evaluate(self.df_test_1)
        mse_2 = self.model.evaluate(self.df_test_2)
        return mse_1, mse_2

    def data_distributions(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Returns a tuple of tuples of floats:
        ((mean_1, std_1), (mean_2, std_2), )
        mean and std of test set 1 (same timeframe as train) and test set 2 (latest data)
        t_stat and p_val from independent 2-sample t test
        """

        X_1 = self.df_test_1['rating'].to_numpy()
        X_2 = self.df_test_2['rating'].to_numpy()

        X_1 = X_1[~np.isnan(X_1)]
        X_2 = X_2[~np.isnan(X_2)]

        mean_1 = np.mean(X_1)
        mean_2 = np.mean(X_2)

        std_1 = np.std(X_1)
        std_2 = np.std(X_2)

        t_stat, p_val = stats.ttest_ind(X_1, X_2, equal_var=False)

        return (mean_1, std_1), (mean_2, std_2), t_stat, p_val

if __name__ == '__main__':
    oe = OfflineEvaluation('svd_model_4_2_0')
    mses = oe.mses()
    data_distrubtions = oe.data_distributions()

    with open('offline_metrics.txt', 'w') as f:
        f.write(f'MSE_main={mses[0]}\n')
        f.write(f'MSE_recent={mses[1]}\n')
        f.write(f'Mean_main={data_distrubtions[0][0]}\n')
        f.write(f'STD_main={data_distrubtions[0][1]}\n')
        f.write(f'Mean_recent={data_distrubtions[1][0]}\n')
        f.write(f'STD_recent={data_distrubtions[1][1]}\n')
        f.write(f't-stat={data_distrubtions[2]}\n')
        f.write(f'p-val={data_distrubtions[3]}')
