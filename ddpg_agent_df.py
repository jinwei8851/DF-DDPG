from collections import deque
import random
import numpy as np
import tensorflow as tf
import tensorlayer as tl


class DDPG:

    def __init__(self, n_features, action_low, action_high, n_hidden, learning_rate_actor,
                 learning_rate_critic, gamma, tau, var, clip_min, clip_max, memory_size, batch_size):
        """
            Args:
                n_features: 状态的维度
                action_low: 动作的最小值
                action_high: 动作的最大值
                n_hidden: 隐藏层神经元数目
                learning_rate_actor: 行动者学习率
                learning_rate_critic: 评论家学习率
                gamma: 折扣因子
                tau: 软更新因子
                var: 输出动作的噪声范围
                clip_min: 裁减值的下边界
                clip_max: 裁减值的上边界
                memory_size: 经验池容量
                batch_size: 一个批次的大小
        """

        # 超参数
        self.n_features = n_features
        self.action_low = action_low
        self.action_high = action_high
        self.n_hidden = n_hidden
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.gamma = gamma
        self.tau = tau
        self.var = var
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=self.memory_size)  # 构建经验池，实质上就是一个双端队列

        # 初始化网络参数
        self.W = tf.random_normal_initializer(mean=0, stddev=0.3)
        self.b = tf.constant_initializer(0.1)

        # 建立行动者网络和目标行动者Q网络
        self.actor = self.buildActor([None, self.n_features])  # 建立行动者网络
        self.actor.train()  # 指定网络用于训练

        self.actor_target = self.buildActor([None, self.n_features])  # 创建目标行动者Q网络
        self.actor_target.eval()  # 指定网络不用于更新

        self.copyPara(self.actor, self.actor_target)  # 后面采用软更新的方式，所以此处需要先复制一遍网络参数

        # 建立评论家网络和目标评论家网络
        self.critic = self.buildCritic([None, self.n_features], [None, 1])  # Critic网络需要s和a的值，a是一维的
        self.critic.train()

        self.critic_target = self.buildCritic([None, self.n_features], [None, 1])
        self.critic_target.eval()

        self.copyPara(self.critic, self.critic_target)

        # 建立优化器
        self.actor_opt = tf.optimizers.Adam(self.learning_rate_actor)
        self.critic_opt = tf.optimizers.Adam(self.learning_rate_critic)

        # 建立ema，滑动平均值，用来更新网络参数
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)  # soft replacement

    def buildActor(self, inputs_shape):
        """
            创建actor网络

            Args:
                inputs_shape: 输入数据的维度
        """
        x = tl.layers.Input(inputs_shape)

        hidden_first = tl.layers.Dense(n_units=self.n_hidden, act=tf.nn.relu,
                                       W_init=tf.initializers.GlorotUniform(), b_init=self.b)(x)
        hidden_second = tl.layers.Dense(n_units=self.n_hidden, act=tf.nn.relu,
                                        W_init=tf.initializers.GlorotUniform(), b_init=self.b)(hidden_first)
        output = tl.layers.Dense(n_units=1, W_init=tf.initializers.GlorotUniform(), b_init=self.b)(hidden_second)

        return tl.models.Model(inputs=x, outputs=output)

    def buildCritic(self, inputs_state_shape, inputs_action_shape):
        """
            建立critic网络，其中动作默认是1维，因为做的是预测

            Args:
                inputs_state_shape: 输入状态的维度
                inputs_action_shape: 输入动作的维度
        """
        s = tl.layers.Input(inputs_state_shape)
        a = tl.layers.Input(inputs_action_shape)
        x = tl.layers.Concat(1)([s, a])

        hidden_first = tl.layers.Dense(n_units=self.n_hidden, act=tf.nn.relu,
                                       W_init=self.W, b_init=self.b)(x)
        hidden_second = tl.layers.Dense(n_units=self.n_hidden,
                                        W_init=self.W, b_init=self.b)(hidden_first)
        y = tl.layers.Dense(n_units=1, W_init=self.W, b_init=self.b)(hidden_second)

        return tl.models.Model(inputs=x, outputs=y)

    def copyPara(self, from_model, to_model):
        """
            更新参数，首次完全复制，后面采用滑动平均的方式进行更新参数的更新

            Args:
                from_model: 参数来源
                to_model: 参数目的地
        """
        for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
            j.assign(i)

    def emaUpdate(self):
        """
            滑动平均更新，其实和之前的硬更新类似，不过在更新赋值之前，用一个ema.average。
        """
        paras = self.actor.trainable_weights + self.critic.trainable_weights  # 获取要更新的参数包括actor和critic的

        self.ema.apply(paras)  # 主要是建立影子参数

        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))  # 用滑动平均赋值

    def updateNetwork(self, state, action, reward, next_state):
        """
            利用数据更新 Actor Critic 网络的参数

            Args:
                state: 状态矩阵，每一行对应一个状态 (batch_size, n_features)
                action: 状态对应的动作 (batch_size, n_actions)
                reward: 即使奖赏 (batch_size, 1)
                next_state: 下一转移状态 (batch_size, n_features)
        """

        # Critic更新
        with tf.GradientTape() as tape:
            next_action = self.actor_target(next_state)  # 计算动作值

            q_next = self.critic_target([next_state, next_action])  # 计算动作值对应的Q值

            q_target = reward + self.gamma * q_next

            q = self.critic([state, action])

            td_error = tf.losses.mean_absolute_error(q_target, q)

        critic_grads = tape.gradient(td_error, self.critic.trainable_weights)

        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

        # Actor 更新，用 Critic 网络的结果去更新actor
        with tf.GradientTape(persistent=True) as tape:
            action_pre = self.actor(state)

            q_pre = self.critic([state, action_pre])

            actor_loss = -tf.reduce_mean(q_pre)  # 梯度上升

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)

        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))

    def choose_action(self, state, scale1, scale2, kind, kind_proba):
        """
            根据状态选择相应的动作

            Args:
                state: 传入的状态
                scale1,scale2: 1 向下 2向上
                kind: 1 向下 2向上

            Return:
                action: 直接返回动作值，不存在动作个数的概念了
        """
        # 调整 state 的输入格式，state输入进来是(7,)，调整为(7,1)
        scale1 = 0.780
        scale2 = 1.333
        state = np.reshape(state, (1, -1))
        state = np.array(state, dtype="float32")
        base_state = state[:,23]


        action = self.actor(state)  # 获取动作值

        if kind == 2 and action >=(1+scale2)*base_state:
            return action

        elif kind == 2 and action <=(1+scale2)*base_state :
            return 2.822*base_state

        elif kind == 1 and action <=(1/(1-scale1))*base_state:
            return action

        elif kind == 1 and action >=(1/(1-scale1))*base_state:
            return 0.458*base_state

        else:
            return action

        # if stage == "train":  # 如果是训练阶段，那么 action 要被裁减
        #     # action = np.clip(np.random.normal(action, self.var), self.clip_min, self.clip_max)
        #     action = np.random.normal(action, self.var)
        # if kind == 2 and action<1.8state::
        # if action<1.8state:
        #     action=1.8state
        # if way == "expection":
        #     if kind != 0:
        #         action = action + action*kind_proba[1] - action*kind_proba[2]
        # elif way == "single_value":
        #     if kind == 1:
        #         action = action - action*kind_proba[1]
        #     elif kind == 2:
        #         action = action + action * kind_proba[2]


    def store_transition(self, state, action, reward, next_state):
        """
            将与环境交互的数据存储到经验池中

            Args:
                state: 传入的状态
                action: 与环境交互时产生的动作
                reward: 得到的即时奖赏
                next_state: 交互后待转移的状态
        """

        # 改变数据的形状,将其转换为1行n列的矩阵数据格式
        state = np.reshape(state, (1, -1))
        next_state = np.reshape(next_state, (1, -1))
        action = np.reshape(action, (1, -1))
        reward = np.reshape(reward, (1, -1))

        transition = np.concatenate((state, action, reward, next_state), axis=1)  # 将其按列拼接，形成1行m列的矩阵

        # self.memory.append(transition)    # 直接存入1行m列，经验池则是张量，例如存入5个1行m列数据，就是 (5,1，m)
        self.memory.append(transition[0])  # 将1行m列转换为m行数据存储进经验池，经验池将其当做当做一个元素，例如存入5个m行数据，就是 (5,m)

    def learn(self, step):
        """
            构建一个会学习的 agent，可以在有数据的情况下进行网络的更新
        """

        if len(self.memory) == self.memory_size:  # 如果经验池满，则选取数据进行训练

            # 1. 准备好每个网络的参数
            if step % 200 == 0:
                self.emaUpdate()
                self.var = self.var * 0.995

            # 2. 选取要输入到网络中的数据，从经验池中随机挑选batch_size个数据，按列切片，每片是一个矩阵
            batch = np.array(random.sample(self.memory, self.batch_size), dtype="float32")
            batch_s = batch[:, :self.n_features]
            batch_a = batch[:, self.n_features:(self.n_features + 1)]
            batch_r = batch[:, (self.n_features + 1):(self.n_features + 2)]
            batch_s_ = batch[:, (self.n_features + 2):(self.n_features * 2 + 2)]

            # 3. 将数据代入网络进行网络的训练
            self.updateNetwork(state=batch_s, action=batch_a, reward=batch_r, next_state=batch_s_)
