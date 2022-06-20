import numpy as np


class Prediction:
    """
    输入：
        状态，动作，回合数，每个回合的步数

    输出：
        预测值，和各项评价指标
    """

    def train(self, method_str, method, way,state, action, max_episodes, max_steps, file_log, kind_prob, state_kinds=None):
        """
            使用训练数据进行方法的agent的训练

            Args:
                method_str: 方法的名称
                method: 传入的agent, ddpg or dqn
                way:对应choose_action用期望还是固定值
                state: 状态数据集
                action: 真实动作数据集
                max_episodes: 回合数
                max_steps: 每个回合的步数
                state_kinds: 状态对应的类别矩阵

            Returns:
                method: 训练好的agent
                episode_mae: 训练过程中每回合的mae
                action_predict: 每步的预测值组成的列表
                action_true: 每步的真实值组成的列表
                episode_step_reward: 每步奖赏组成的列表
        """

        episode_total_reward = []  # 记录回合奖赏
        episode_step_reward = []  # 记录每步奖赏
        episode_mae = []  # 记录每回合的MAE值，观察是否收敛
        action_true = []  # 记录每步的真实能耗值
        action_predict = []  # 记录每步的预测能耗值


        for episode in range(max_episodes):  # 回合开始

            episode_reward = 0  # 回合的总奖赏初始化
            count_step = 0  # 记录每回合的步骤数

            index = np.random.choice(range(len(state)-1))  # 随机生成索引 0 - len(state)-1
            s = state[index]  # 选择该索引对应的时刻的各项属性值作为状态


            if method_str == "DF-DQN" or method_str == "DDPG":
                kind = state_kinds[index]

            for step in range(max_steps):
                count_step += 1  # 记录每回合的步骤数

                a_true = action[index]  # 根据状态s的索引，取出对应的真实动作值
                action_true.append(a_true)  # 保存真实动作值a_true

                if method_str == "DQN":

                    # 根据状态s选择动作a，注意区分动作和动作值，看需要哪个
                    a, a_value = method.choose_action(state=s, stage="train")
                    action_predict.append(a_value)  # 保存预测动作值a
                    r = -abs(a_value - a_true)  # 计算即时奖赏

                elif method_str == "DF-DQN":
                    # 根据状态类别，选择合适的动作空间，再选择合适的动作
                    a, a_value = method.choose_action(state=s, kind=kind, stage="train")
                    action_predict.append(a_value)  # 保存预测动作值a
                    r = -abs(a_value - a_true)  # 计算即时奖赏

                elif method_str == "DDPG":
                    a = method.choose_action(state=s, kind=kind,way="single_value",stage="train",kind_proba=kind_prob)  # 根据状态s选择动作a
                    a = np.reshape(a, (1, -1))[0][0]
                    action_predict.append(a)  # 保存预测动作值a
                    r = -abs(a - a_true)  # 计算即时奖赏

                episode_step_reward.append(r)  # 记录每步骤的即时奖赏

                episode_reward += r  # 计算一个回合的累计奖赏

                index += 1  # 移动至下一状态的索引

                if index == len(state):  # 如果下一状态索引已经超出状态数组范围，则进入下个回合
                    break

                s_ = state[index]  # 保存下一状态

                if method_str == "DF-DQN" or method_str == "DDPG":
                    kind = state_kinds[index]

                method.store_transition(s, a, r, s_)  # 将这次与环境交互存入到样本池中

                method.learn(count_step)  # 让agent进行学习

                # 如果状态数组遍历完毕，或者达到步数（遍历数组的要求）
                if (index == len(state) - 1) or (step == max_steps - 1):
                    episode_total_reward.append(episode_reward)  # 保存每个回合的累计奖赏
                    print('Episode %d : %.2f' % (episode, episode_reward))
                    file_log.write('Episode %d : %.2f\n' % (episode, episode_reward))  # 打印回合数和奖赏累计值
                    break

                s = s_  # 如果这个回合没结束，那就顺着时序采样，继续训练
            episode_reward = np.reshape(episode_reward, (1, -1))[0][0]
            episode_mae.append((-episode_reward) / count_step)  # 计算每回合的mae，看收敛情况

        return method, episode_mae, action_predict, action_true, episode_step_reward  # 将训练好的agent返回出去

    def prediction(self, method_str, method, state, action, kind_prob,state_kinds=None):
        """
            用已经训练好的agent去预测

            Args:
                method_str: 方法的名称
                method: agent的方法名称
                state: 状态数据集
                action: 真实动作集
                state_kinds: 先加上这个参数，后期应该调用网络进行分类

            Returns:
                action_predict: 预测值组成的列表
                action_true: 真实值组成的列表
        """

        action_predict = []  # 测试集选择的每个动作，也就是测试集所有的预测值
        action_true = []  # 测试集的标签

        for i in range(len(state)):

            s = state[i]  # 直接从测试集中取出状态

            if method_str == "DF-DQN" or method_str == "DDPG":
                kind = state_kinds[i]

            action_true.append(action[i])  # 保存真实动作值

            if method_str == "DQN":
                a, a_value = method.choose_action(state=s, stage="test")  # 根据状态选择动作
                action_predict.append(a_value)  # 保存预测动作值

            elif method_str == "DF-DQN":
                a, a_value = method.choose_action(state=s, kind=kind, stage="test")  # 根据状态选择动作
                action_predict.append(a_value)  # 保存预测动作值

            elif method_str == "DDPG":
                a = method.choose_action(state=s, stage="test",kind=kind,way="single_value",kind_proba=kind_prob)
                a = np.reshape(a, (1, -1))[0][0]    # 网络直接输出是张量, 将其转换成一个数值
                action_predict.append(a)  # 保存预测动作值

        return action_predict, action_true
