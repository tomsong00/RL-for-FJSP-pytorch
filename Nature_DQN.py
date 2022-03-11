"""summary
DQN步骤：

- 记忆池里的数据样式
- CartPole-v0的状态由4位实数编码表示，所以第一层网络是4->50
"""
# %%


import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Machine
import Task
from Reform_task import reform_task
from Instantiate_task import instantiate_task
from State_transform import state_transform
import Scheduling
from numpy.lib.function_base import _quantile_dispatcher
# from thop import profile
from Gantt_graph import Gantt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import gym
from matplotlib import pyplot as plt
import sys

# Hyper Parameters
BATCH_SIZE = 8
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.4                 # reward discount
TARGET_REPLACE_ITER = 7   # target update frequency
MEMORY_CAPACITY = 512
DEVICE = 0   # 指定GPU
# env = gym.make('CartPole-v0')
# env = env.unwrapped
# N_ACTIONS = 4  # 4种候选的算子
# N_STATES = 30  # 30维决策变量
use_gpu = torch.cuda.is_available()
# %%


class Net(nn.Module):
    def __init__(self, inDim, outDim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inDim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.out = nn.Linear(32, outDim)
        #self.pool=nn.AdaptiveAvgPool2d((1,outDim))
    def forward(self, x):
        # return self.out(F.relu(self.fc1(x)))
        #x=torch.flatten(x)
        x = F.softmax(x, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x= self.out(x)
        #x= self.pool(x)
        action_value =x
        return action_value


class DQN(object):
    def __init__(self, inDim, outDim):
        self.eval_net, self.target_net = Net(inDim, outDim), Net(inDim, outDim)
        # global N_STATES, N_ACTIONS
        self.N_STATES = inDim
        self.N_ACTIONS = outDim
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        # self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        # memory是一个np数组，每一行代表一个记录，状态 动作 奖励 新的状态
        self.memory = np.zeros((MEMORY_CAPACITY, self.N_ACTIONS*self.N_STATES *2 + 2))     # initialize memory
        self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        if use_gpu:
            self.eval_net, self.target_net = self.eval_net.cuda(DEVICE), self.target_net.cuda(DEVICE)
            self.loss_func = self.loss_func.cuda(DEVICE)

    def choose_action(self, x):
        # x: a game state
        # 在前面多加一维，可能是一批数据的意思
        # 返回的是0-1动作整数编码
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if use_gpu:
            x = x.cuda(DEVICE)
        # flops, params = profile(self.eval_net, inputs=x)
        # print(flops, params)
        # input only one sample
        # if np.random.uniform() < EPSILON:   # greedy
        if np.random.uniform() < 2:   # greedy
            actions_value = self.eval_net.forward(x)  # shape=(1,action)
            #想要保存要从cpu转换
            if use_gpu:
                actions_value = actions_value.cpu()
            actions_value = actions_value.detach().numpy()
            # print(actions_value)
            # action = np.argmax(actions_value[0])  # 选择回报最大的动作
            # print(action)
            # return action
            actions_value[actions_value <= 0] = 0.001  # 不能有负概率
            # actions_value = actions_value / np.sum(actions_value)  # 归一化
            # 计算排名
            #argsort_ = self.N_ACTIONS - 1 - np.argsort(np.argsort(actions_value[0]))
            # 以系数c拉大概率差距
            #c = 0.5
            #for i in range(self.N_ACTIONS):
                #actions_value[0][i] = actions_value[0][i] * c**argsort_[i]
            # 手动设计概率，和排名拼接
            #probability_value = np.array([[70, 28, 10, 8, 5, 5]])
            #probability_value = probability_value / np.sum(probability_value)
            #actions_value = probability_value[:, argsort_]

            #actions_value = actions_value / np.sum(actions_value)

            # 按照概率取样

            try:
                #action = np.random.choice(self.N_ACTIONS, size=1, p=actions_value[0])[0]
                action=np.argmax(actions_value,axis = 1)
                action=np.argmax(action)
                #action = np.random.randint(0, self.N_ACTIONS)
            except:
                #print(actions_value)
                action = np.random.randint(0, self.N_ACTIONS)
            # print(actions_value, action)
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.N_ACTIONS)   # [0,N_ACTIONS)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        # 数组合并，a和r也新建个数组
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self,BATCH_SIZE):
        # target parameter update
        # 每隔一定步骤，更新target net
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # TODO  减去baseline
        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        #状态 动作 奖励 新的状态，self.N_STATES * 2 + 2
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES + 1].astype(int))  # 动作是int型
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES + 1:self.N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])
        #b_s=b_s.view([-1,BATCH_SIZE,self.N_STATES])
        if use_gpu:
            b_s = b_s.cuda(DEVICE)
            b_a = b_a.cuda(DEVICE)
            b_r = b_r.cuda(DEVICE)
            b_s_ = b_s_.cuda(DEVICE)

        #计算两个网络的目标值，需要改
        # q_eval w.r.t the action in experience
        #b_s=np.reshape((-1,))
        q_eval = self.eval_net(b_s)
        #q_eval = q_eval.view([1, self.N_ACTIONS])
        q_eval=q_eval.gather(1, b_a)  # shape (batch, 1)
        #q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        # q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        q_target = b_r
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print("损失值为{}".format(loss.item()))


if __name__ == '__main__':
    # 参数设置-----------------------------------------------------------------------
    scheduling = Scheduling.Scheudling()
    machine = Machine.Machine()
    task = Task.Task()
    task_set = task.load_task(1, 1)

    machine_set = machine.load_machine(1, 1)
    machine_num = np.shape(machine_set)[0]
    task_num = np.shape(task_set)[0]
    temp = np.zeros([task_num, 1])
    #task_set = reform_task(task_set)
    #task_set = instantiate_task(task_set)
    state_task_set = np.hstack((task_set,temp))
    state_num=np.shape(state_task_set)[1]
    N_ACTIONS = task_num
    Epoch=200
    N_STATES = state_num
    BATCH_SIZE = 16
    MEMORY_CAPACITY = 40
    CountOpers = np.zeros(N_ACTIONS)
    PopCountOpers = []
    dqn = DQN(N_STATES, N_ACTIONS)
    total_fitness=np.zeros(Epoch)
    s = state_task_set
    s_ = state_task_set
    task_size = np.shape(task_set)[0]
    task_set = reform_task(task_set)
    print('\nCollecting experience...')
    for i_episode in range(0, Epoch):
        action_choosen = np.zeros((1,task_num))
        processor = np.zeros([int(task_set[task_size - 1][1]), 3])  # 任务前序任务编号和完成时间和完成机器编号，行=大任务数量
        #total_fitness[i_episode] = 0
        for t in range(N_ACTIONS):
            t_start=time.time()
            a = dqn.choose_action(s)
            a = int(a)
            #print(t)
            #qianxu_neibubianhao = task_set[a][2]
            while action_choosen[0][a] == 1:
               a = np.random.randint(0, task_num)
            # 0:id  1:所属大任务  2:内部编号  3:允许执行的机器类型编号  4:准备时间  5:加工时间 6:机器编号
            # darenwu_a = np.int(task_set[a][1]) #当前a的大任务编号
            # qianxurenwu =np.int(processor[np.int(darenwu_a-1)][0])
            # if np.int(task_set[qianxurenwu-1][2]+1) != np.int(task_set[a][2]):
            #     a = np.int(task_set[qianxurenwu+1-1-1][0]-1)
                # if task_set[qianxurenwu-1][2]+1 != task_set[a][2] and qianxurenwu != 0:
                #     a = np.random.randint(1, task_num+1)
                # if qianxurenwu == 0:
                #     a = np.random.randint(1, task_num+1)
            #print("action:"+str(a))
            action_choosen[0][a] = 1


            s_ = state_transform(s,s_,a)
            information_network = np.ones([machine_num, machine_num])

            temp_a = a + 1

            action = np.reshape(temp_a,(-1,1))

            if t==0:
                scheduling_result,temp_machine_set,processor = scheduling.arrange_task_to_machine_RL(action, task_set, machine_set,
                                                                   information_network,processor)
                lie = np.shape(scheduling_result[0])[1]
                temp_result = np.reshape(scheduling_result[0][0, :], (-1, lie))
                temp_result = np.array(temp_result, dtype='int64')
                temp_schedualing_result = np.zeros([1,lie])
                for b in range(lie):
                    temp_schedualing_result[0][b] = int(temp_result[0][b])
            if t>0:
                scheduling_result,temp_machine_set,processor = scheduling.arrange_task_to_machine_RL(action, task_set, temp_machine_set,
                                                                          information_network,processor)
                if scheduling_result[0][0,0]!=0:
                    lie = np.shape(scheduling_result[0])[1]
                    temp_result = np.reshape(scheduling_result[0][0,:],(-1,lie))
                    temp_result = np.array(temp_result,dtype='int64')

                    #temp_schedualing_result = temp_result
                    # for b in range(lie):
                    #     temp_schedualing_result[0][b] = np.int(temp_result[0][b])
                    temp_schedualing_result=np.vstack((temp_schedualing_result,temp_result))
                    temp_schedualing_result = np.array(temp_schedualing_result,dtype='int64')
            fitness = scheduling.fitness_calculation_RL(scheduling_result, action)

            r = fitness[0]
            total_fitness[i_episode]=total_fitness[i_episode]+r
            #s = s.flatten()
            s_temp = s_.flatten()
            dqn.store_transition(s.flatten(), a, r, s_temp)
            if dqn.memory_counter > 50:
                dqn.learn(BATCH_SIZE)
            CountOpers[a] += 1
            if i_episode % 5 == 0:
                #print(i_episode, ' ', a)
                PopCountOpers.append(CountOpers)
                CountOpers = np.zeros(N_ACTIONS)
            s = s_

        #print(temp_schedualing_result)
        t_end=time.time()
        print("时间为{}s".format(t_end-t_start))

    torch.save(dqn.eval_net.state_dict(), './model_dic/nn_{}.pth'.format(i_episode+1))
    Gantt(temp_schedualing_result)
    PopCountOpers = np.array(PopCountOpers)
    for i in range(N_ACTIONS):
        plt.plot(PopCountOpers[:, i], '.', label=str(i))
    plt.legend()
    plt.show()

    sys.exit(0)


"""
    while True:
        # env.render()
        a = dqn.choose_action(s)

        # take action
        # s_, r, done, info = env.step(a)
        s_ = np.random.uniform(-1,1,N_STATES)

        # modify the reward
        # x, x_dot, theta, theta_dot = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = r1 + r2
        # r = a
        r = -(np.abs(a-2))
        r += 2
        r = np.random.normal(r,3,1)[0]

        dqn.store_transition(s, a, r, s_)
        # print(s,a,r,s_)
        # sys.exit(0)
        line = 20

        ep_r += r
        # if dqn.memory_counter > MEMORY_CAPACITY:
        if dqn.memory_counter > 20:
            dqn.learn()
            # print(a)
            # if ep_r>line:
            #     print('Ep: ', i_episode,
            #           '| Ep_r: ', round(ep_r, 2))

        if ep_r>line:
            break
        s = s_

"""
