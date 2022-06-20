# coding=gbk
import os
import time
import shutil
import numpy as np
import pandas as pd
from tool import Tool
from dqn_agent import DQN
from ddpg_agent_df import DDPG
from prediction_new import Prediction
from draw_picture import DrawPicture
from create_samples import CreateSample
from state_classify import StateClassify
from df_dqn_agent import DF_DQN
from utils import utils

dir = os.getcwd()  # ��ĿĿ¼
dir_data = dir + "/data/"  # ����Ŀ¼
data = pd.read_csv(dir_data + 'index_3_W (2015.1.1-2016.12.31).csv', header=None)
# newcolumns = ['energy', 't1', 'wind']
# data.columns = newcolumns
data = data.iloc[:, [1]]
data = np.array(data)
scale1 = 0.785
scale2 = 1.384
point = utils().generatepoints(data=data,shape="percentage",features=0,scale1=scale1,scale2=scale2)

"""�ϲ���ָ���Լ�"""
features = 24
data, label = CreateSample().createFirstSample(data=data, shape="vector", features=features)

# �ָ�ѵ�����Ͳ��Լ�����
data_train = np.array(data[0:len(data) - 61 * 24])  # 2015��1����2016��10��
label_train = np.array(label[0:len(label) - 61 * 24])

data_test = np.array(data[len(data) - 61 * 24:])  # 2016��11�£�12��
label_test = np.array(label[len(label) - 61 * 24:])

data_train_min = np.min(data_train)  # ѵ�����е�����ܺ�
data_train_max = np.max(data_train)  # ѵ�����е�����ܺ�

# filename_log = ""
# data_train, data_test = Tool(filename_log).normalization(data_train=data_train, data_test=data_test)

date_data = pd.read_csv(dir_data + 'test3W_new.csv', header=0, skiprows=23, nrows=16032)
date_data = date_data.iloc[:, [ 7,8, 9, 10, 11, 12,13]]
date_data2 = pd.read_csv(dir_data + 'test3W_new.csv', header=0, skiprows=16055, nrows=1464)
date_data2 = date_data2.iloc[:, [7,8, 9, 10, 11, 12,13]]
#
data_train = np.hstack((data_train, date_data))
data_test = np.hstack((data_test, date_data2))

data_trainlabel = point[23:16055]
data_trainlabel = np.reshape(np.array(data_trainlabel),(-1,1))
# data_train = np.hstack((data_train, data_trainlabel))


data_testlabel = point[16055:17519]
data_testlabel = np.reshape(np.array(data_testlabel),(-1,1))
# data_test = np.hstack((data_test, data_testlabel))

state_train_scale, state_test_scale, class_train_pre, class_test_pre,class_train_proba,class_test_proba,acc_test= utils().constructState(data_train_scale=data_train,data_test_scale=data_test,
                                                                                              class_train_true=data_trainlabel,class_test_true=data_testlabel)
print(state_train_scale.shape,state_test_scale.shape)

# # 1.3 ��׼��
# filename_log = ""
# data_train_scale, data_test_scale = Tool(filename_log).normalization(data_train=state_train_scale, data_test=state_test_scale)

MAX_EPISODES = 200  # ѵ�������غ���
MAX_STEPS = 1000  # ÿ���غϵ������
CLASS = 8  # ��ǰ�����
N_CLASS = 10  # ״̬�������
iteration = 8  # ʵ���������


METHOD_STR = "DDPG"  # DQN��DF-DQN��DDPG
dir_dqn = dir + '\\experiments\\DQN' + '\\DQN_index='  # DQN��������·��
dir_ddpg = dir + '\\experiments\\DF-DDPG(1.75sigma)' + '\\DF-DDPG(1.75sigma)_index='  # DDPG��������·��

while True:

    CLASS = CLASS + 1

    dir_dfdqn = dir + '\\experiments\\DF-DQN' + '\\N_Class=' + str(CLASS) + '\\DF-DQN_index='

    for index in np.arange(0, 10):

        """
            ��һ�׶γ���ʼ����
        """
        start_first = time.perf_counter()

        # determine the path of methods
        if METHOD_STR == "DF-DQN":
            dir_choose = dir_dfdqn + str(index)

        elif METHOD_STR == "DQN":
            dir_choose = dir_dqn + str(index)

        elif METHOD_STR == "DDPG":
            dir_choose = dir_ddpg + str(index)

        if os.path.exists(dir_choose):
            shutil.rmtree(dir_choose)
        os.makedirs(dir_choose)

        # ������־����¼�������
        filename_log = "log.txt"
        file_log = open(dir_choose + "\\" + filename_log, 'w')
        file_log.write("�����½��ٷֱ� " + str(scale1) + " ���������ٷֱ�  " + str(scale2)  + "\n")
        file_log.write("acc_test: {:.3f} %\n\n".format(acc_test))

        # 2.1 DF-DQN���������ܺ�Ԥ��
        if METHOD_STR == "DF-DQN":

            # ״̬����
            gap = np.ceil((data_train_max - data_train_min + 1) / CLASS)  # ÿ������ļ��

            print("���� ", str(CLASS), " �������£�", "�� ", str(index), " ��ѭ��")
            print("�����ռ仮�� ", str(CLASS), " ��", "���ܶ�������Ϊ ", str(gap), " ��")

            file_log.write("���� " + str(CLASS) + " �������£�" + "�� " + str(index) + " ��ѭ��" + "\n")
            file_log.write("�����ռ仮�� " + str(CLASS) + " ��" + "���ܶ�������Ϊ " + str(gap) + " ��\n\n")

            class_train_true = ((label_train - data_train_min) / gap).astype(int)  # ѵ����״̬������ʵֵ
            class_test_true = ((label_test - data_train_min) / gap).astype(int)  # ���Լ�״̬������ʵֵ

            # ʹ�����ɭ�ֵõ�״̬��Ӧ��������
            state_train_scale, state_test_scale, class_train_pre, class_test_pre = StateClassify().constructState(
                data_train_scale=data_train_scale, data_test_scale=data_test_scale,
                class_train_true=class_train_true, class_test_true=class_test_true, file_log=file_log)

            """
                ��һ�׶γ������н���
            """
            end_first = time.perf_counter()

            """
                �ڶ��׶γ���ʼ����
            """
            start_second = time.perf_counter()

            # 4.2 DF-DQN�ܺ�Ԥ��
            # DF-DQN�Ĳ�������
            N_FEATURES = features + CLASS+ 5
            ACTION_START = data_train_min
            ACTION_END = data_train_min + gap
            N_ACTIONS = int(gap)  # Ĭ�ϼ��Ϊ1
            N_HIDDEN = 32
            LEARNING_RATE = 0.01
            GAMMA = 0.9
            EPSILON = 0.5
            EPSILON_DECAY = 0.995
            EPSILON_MIN = 0.01
            MEMORY_SIZE = 2000
            BATCH_SIZE = 64

            df_dqn = DF_DQN(n_features=N_FEATURES, n_class=CLASS, action_start=ACTION_START,
                            action_end=ACTION_END,
                            n_actions=N_ACTIONS, n_hidden=N_HIDDEN, learning_rate=LEARNING_RATE, gamma=GAMMA,
                            epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN,
                            memory_size=MEMORY_SIZE,
                            batch_size=BATCH_SIZE)

            dqn_train, mae_train, predict_train, actual_train, reward_train = Prediction(). \
                train(method_str=METHOD_STR, method=df_dqn, state=state_train_scale, action=label_train,
                      max_episodes=MAX_EPISODES, max_steps=MAX_STEPS, file_log=file_log, state_kinds=class_train_pre)

            predict_test, actual_test = Prediction().prediction(
                method_str="DF-DQN", method=dqn_train, state=state_test_scale, action=label_test,
                state_kinds=class_test_pre)

        elif METHOD_STR == "DQN":

            print("�� ", str(index), " ��ѭ��")
            file_log.write("�� " + str(index) + " ��ѭ��" + "\n")

            """
                ��һ�׶γ������н���
            """
            end_first = time.perf_counter()

            """
                �ڶ��׶γ���ʼ����
            """
            start_second = time.perf_counter()

            N_FEATURES = features
            N_ACTIONS = int(data_train_max - data_train_min + 1)
            ACTION_LOW = data_train_min
            ACTION_HIGH = data_train_max
            N_HIDDEN = 32
            LEARNING_RATE = 0.01
            GAMMA = 0.9
            EPSILON = 0.5
            EPSILON_DECAY = 0.995
            EPSILON_MIN = 0.01
            MEMORY_SIZE = 2000
            BATCH_SIZE = 64

            dqn = DQN(n_features=N_FEATURES, n_actions=N_ACTIONS, n_hidden=N_HIDDEN, action_low=ACTION_LOW,
                      action_high=ACTION_HIGH,
                      learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON, epsilon_decay=EPSILON_DECAY,
                      epsilon_min=EPSILON_MIN, memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE)

            dqn_train, mae_train, predict_train, actual_train, reward_train = Prediction().train(
                method_str=METHOD_STR, method=dqn, state=data_train_scale,
                action=label_train, max_episodes=MAX_EPISODES, max_steps=MAX_STEPS, file_log=file_log)

            predict_test, actual_test = Prediction().prediction(
                method_str="DQN", method=dqn_train, state=data_test_scale, action=label_test)

        elif METHOD_STR == "DDPG":

            print("�� ", str(index), " ��ѭ��")
            file_log.write("�� " + str(index) + " ��ѭ��" + "\n")

            """
                ��һ�׶γ������н���
            """
            end_first = time.perf_counter()

            """
                �ڶ��׶γ���ʼ����
            """
            start_second = time.perf_counter()

            N_FEATURES = features
            ACTION_LOW = data_train_min
            ACTION_HIGH = data_train_max
            CLIP_MIN = data_train_min
            CLIP_MAX = data_train_max
            N_HIDDEN = 32
            LEARNING_RATE_ACTOR = 0.001
            LEARNING_RATE_CRITIC = 0.001
            GAMMA = 0.9
            TAU = 0.1
            VAR = 40
            MEMORY_SIZE = 2000
            BATCH_SIZE = 64
            WAY = "expection"

            ddpg = DDPG(n_features=34, action_low=ACTION_LOW, action_high=ACTION_HIGH, n_hidden=N_HIDDEN,
                        learning_rate_actor=LEARNING_RATE_ACTOR, learning_rate_critic=LEARNING_RATE_CRITIC,
                        gamma=GAMMA, tau=TAU, var=VAR, clip_min=CLIP_MIN, clip_max=CLIP_MAX, memory_size=MEMORY_SIZE,
                        batch_size=BATCH_SIZE)

            ddpg_train, mae_train, predict_train, actual_train, reward_train = Prediction().train(
                method_str=METHOD_STR, method=ddpg, way=WAY,state=state_train_scale, action=label_train,
                max_episodes=MAX_EPISODES, max_steps=MAX_STEPS, file_log=file_log,kind_prob=class_train_proba,state_kinds=class_train_pre)

            predict_test, actual_test = Prediction().prediction(
                method_str=METHOD_STR, method=ddpg_train, state=state_test_scale, action=label_test,kind_prob=class_test_proba,state_kinds=class_test_pre)

        """
            �ڶ��׶γ������н���
        """
        end_second = time.perf_counter()

        print("\n��һ�׶�ʱ��Ϊ��", str(end_first - start_first))
        file_log.write("\n��һ�׶�ʱ��Ϊ��" + str(end_first - start_first) + '\n')

        print(str(MAX_EPISODES), "�غϣ�ÿ���غ�", str(MAX_STEPS), "���������")
        print("�ڶ��׶�ʱ��Ϊ��", str(end_second - start_second))

        file_log.write(str(MAX_EPISODES) + "�غϣ�ÿ���غ�" + str(MAX_STEPS) + "���������\n")
        file_log.write("�ڶ��׶�ʱ��Ϊ��" + str(end_second - start_second) + "\n")

        print("��ʱ��Ϊ��", str(end_first - start_first + end_second - start_second))
        file_log.write("��ʱ��Ϊ��" + str(end_first - start_first + end_second - start_second) + '\n')

        # 5. �����е����ݽ��б��棬�ܹ����������ļ� mae_train, predict_train, actual_train, reward_train, predict_test, actual_test
        save_data_list = [mae_train, predict_train, actual_train, reward_train, predict_test, actual_test]
        save_data_filename = ["mae_train.csv", "predict_train.csv", "actual_train.csv", "reward_train.csv",
                              "predict_test.csv", "actual_test.csv"]

        for j in range(len(save_data_list)):

            data_temp = pd.DataFrame(save_data_list[j])

            data_temp_filename = "\\" + save_data_filename[j]

            if os.path.exists(dir_choose + data_temp_filename):  # �ļ�������ɾ���������ھ�д��
                os.remove(dir_choose + data_temp_filename)

            data_temp.to_csv(dir_choose + data_temp_filename, index=False, header=None)

        # 6. ����ָ��
        print("ѵ��������ָ�꣺")
        file_log.write("\nѵ��������ָ�꣺\n")
        Tool(file_log).mae(action_predict=predict_train, action_true=actual_train)
        Tool(file_log).rmse(action_predict=predict_train, action_true=actual_train)
        Tool(file_log).mape(action_predict=predict_train, action_true=actual_train)
        Tool(file_log).r2(action_predict=predict_train, action_true=actual_train)

        print("====================================================================")
        print("���Լ�����ָ�꣺")
        file_log.write("====================================================================\n")
        file_log.write("���Լ�����ָ�꣺\n")
        Tool(file_log).mae(action_predict=predict_test, action_true=actual_test)
        Tool(file_log).rmse(action_predict=predict_test, action_true=actual_test)
        Tool(file_log).mape(action_predict=predict_test, action_true=actual_test)
        Tool(file_log).r2(action_predict=predict_test, action_true=actual_test)
        file_log.write("�¹����з�Χ��1.7145750277188447��-1.0346283883747236�ж�,10-14\n")
        # 7. ��ͼ
        # ѵ������ͼ
        DrawPicture().Xrange_Y(dir=dir_choose, figName="mae_train", Yname="mae", Y=mae_train)

        DrawPicture().Xthousand_Ypredicted_Yactual(dir=dir_choose, figName="train_compare",
                                                   action_predict=predict_train, action_true=actual_train)

        # ���Լ���ͼ
        DrawPicture().Xpredicted_Yactual(dir=dir_choose, figName="trend",
                                         action_predict=predict_test, action_true=actual_test)

        DrawPicture().Xrange_Ypredicted_Yactual(dir=dir_choose, figName="predict and actual in test set",
                                                action_predict=predict_test, action_true=actual_test)

        DrawPicture().Xthousand_Ypredicted_Yactual(dir=dir_choose, figName="test_compare",
                                                   action_predict=predict_test, action_true=actual_test)

        # �������仯ͼ
        DrawPicture().XrangeError(dir=dir_choose, figName="test_error", action_predict=predict_test,
                                  action_true=actual_test)

        # �ر���־
        file_log.close()

    # �����DQN��DDPG������ֱ��������ѭ��
    if METHOD_STR == "DQN" or METHOD_STR == "DDPG":
        break
    elif METHOD_STR == "DF-DQN":
        if CLASS == N_CLASS:  # ����1��ʼ
            break