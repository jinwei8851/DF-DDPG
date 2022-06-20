import numpy as np

class CreateSample:
    """
    输入：
        index_0_Q (2015.1.1-2017.1.31).csv
        由能耗、天气、日期组成的矩阵

    输出：
        第一种样本池 前 features 小时的能耗构成一个样本
    """

    def createFirstSample(self, data, shape, features):
        """
            构建第一种样本池：前 time 小时预测下一小时的能耗

            Args:
                data: 待构建样本池的数据，矩阵形式
                shape: 单个样本是向量（能耗数据）还是矩阵（能耗数据，天气，日期）

            Returns:
                samples_time: 预测样本的列表
                targets_time: 预测样本对应的真实值标签列表
        """
        print("********* 使用第一样本池 *********")
        print("历史能耗数据构成的特征数为：", features)

        samples_time = []
        targets_time = []

        for j in np.arange(features, len(data), 1):

            sample_temp = []  # 帮助构建样本

            for i in range(j - features, j, 1):

                if shape == "vector":
                    sample_temp.append(data[i][0])
                elif shape == "matrix":
                    sample_temp.append(data[i])

            samples_time.append(sample_temp)
            targets_time.append(data[j][0])

        return samples_time, targets_time
    def createSecondSample(self, data, shape):
        """
            构建第二种样本池 前 day 天相同时刻的能耗值作为样本（不看日期性质）

            Args:
                data: 待构建样本池的数据，矩阵形式
                shape: 单个样本是向量（能耗数据）还是矩阵（能耗数据，天气，日期）

            Return:
                samples_day: 预测样本的列表
                targets_day: 预测样本对应的真实值标签列表
        """
        day = 7
        samples_day = []
        targets_day = []

        for j in np.arange(day * 24, len(data), 1):

            sample_temp = []  # 帮助构建样本

            for i in range(j - 24 * day, j, 24):

                if shape == "vector":
                    sample_temp.append(data[i][0])
                elif shape == "matrix":
                    sample_temp.append(data[i])

            samples_day.append(sample_temp)
            targets_day.append(data[j][0])

        return samples_day, targets_day

    def createThirdSample(self, data, shape, features = 5):
        """
            构建第三种样本池 前 features 天 相同日期性质 相同时刻的能耗值作为样本

            Args:
                data: 待构建样本池的数据，矩阵形式
                shape: 单个样本是向量（能耗数据）还是矩阵（能耗数据，天气，日期）

            Return:
                samples_date: 预测样本的列表
                targets_date: 预测样本对应的真实值标签列表
        """
        print("********* 使用第三样本池 *********")
        samples_date = []
        targets_date = []

        for j in np.arange(0, len(data), 1):

            sample_temp = []  # 帮助构建样本

            indices = []  # 每个样本要留存的 date 个 相同日期性质且相同时刻的能耗值

            temp = 1
            while len(indices) < features:

                if j - 24 * temp < 0:
                    break
                elif data[j][-1] == data[j - 24 * temp][-1]:
                    indices.append(j - 24 * temp)

                temp = temp + 1

            # print(indices)

            # 如果len(indices)<features 那就说明构成不了一个样本，那就 continue，进行下个样本的构建
            if len(indices) == features:

                for i in np.arange(features - 1, -1, -1):  # 左闭右开 [6,5,4,3,2,1,0]

                    if shape == "vector":
                        sample_temp.append(data[indices][i][0])
                    elif shape == "matrix":
                        # sample_temp.append(data[indices][i][:-1])
                        sample_temp.append(data[indices][i])

                samples_date.append(sample_temp)
                targets_date.append(data[j][0])

        return samples_date, targets_date