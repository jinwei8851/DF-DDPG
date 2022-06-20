import math
import numpy as np

class Tool:

    def __init__(self, file_log):
        self.file_log = file_log

    def normalization(self, data_train, data_test):
        """
            归一化

            Args:
                data_train: training set
                data_test: test set
            Return:
                normalized data
        """

        mean_train = np.mean(data_train, axis=0)  # 每列取均值
        std_train = np.std(data_train, axis=0)  # 每列取标准差

        data_train_scale = (data_train - mean_train) / std_train
        data_test_scale = (data_test - mean_train) / std_train

        return data_train_scale, data_test_scale

    def mae(self, action_predict, action_true):
        """
            计算MAE
        """
        N = len(action_true)  # 测试集样本数
        sum = 0  # 计算求和

        for i in range(N):  # 计算 |真实值-预测值|的总和
            sum = sum + abs(action_true[i] - action_predict[i])

        MAE = sum / N

        print("评价指标 MAE 的值为：", str(MAE))
        self.file_log.write("评价指标 MAE 的值为：" + str(MAE) + '\n')

    def rmse(self, action_predict, action_true):
        """
            计算RMSE
        """
        sum = 0
        N = len(action_true)  # 测试集样本数

        for i in range(N):
            sum = sum + math.pow(action_true[i] - action_predict[i], 2)

        RMSE = math.sqrt(sum / N)

        print("评价指标 RMSE 的值为：", str(RMSE))
        self.file_log.write("评价指标 RMSE 的值为：" + str(RMSE) + '\n')

    def cv(self, action_predict, action_true):
        """
            计算CV
        """
        sum = 0
        N = len(action_true)  # 测试集样本数

        for i in range(N):
            sum = sum + math.pow(action_true[i] - action_predict[i], 2)

        RMSE = math.sqrt(sum / N)

        sum = 0
        for i in range(N):
            sum = sum + action_true[i]

        actual_mean = sum / N  # 求真实值的平均值

        CV = RMSE / actual_mean

        print("评价指标 CV 的值为：", str(CV))
        self.file_log.write("评价指标 CV 的值为：" + str(CV) + '\n')

    def r2(self, action_predict, action_true):
        """
            计算R^2
        """
        sum_molecule = 0
        sum_denominator = 0

        N = len(action_true)  # 测试集样本数
        sum = 0
        for i in range(N):
            sum = sum + action_true[i]
        actual_mean = sum / N  # 求真实值的平均值

        for i in range(N):
            sum_molecule = sum_molecule + math.pow(action_predict[i] - action_true[i], 2)  # 分子求和
            sum_denominator = sum_denominator + math.pow(action_true[i] - actual_mean, 2)  # 分母求和

        print("评价指标 R^2 的值：", str(1 - sum_molecule / sum_denominator))
        self.file_log.write("评价指标 R^2 的值：" + str(1 - sum_molecule / sum_denominator) + '\n')

    def mape(self, action_predict, action_true):

        """
            计算 MAPE
        """
        N = len(action_true)  # 测试集样本数
        sum = 0  # 计算求和

        for i in range(N):  # 计算 |真实值-预测值|的总和
            sum = sum + abs((action_true[i] - action_predict[i])/action_true[i])

        MAPE = 100 * sum / N

        print("评价指标 MAPE(%) 的值为：", str(MAPE), "%")
        self.file_log.write("评价指标 MAPE(%) 的值为：" + str(MAPE) + " %" + '\n')