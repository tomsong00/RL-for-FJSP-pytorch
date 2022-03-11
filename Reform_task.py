import numpy as np
from Priority import priority
from Opr_num import opr_num
from Choose_graph import choose_graph
import Machine
import Task

#函数返回每一个子任务的优先级
# O[i][0]表示第i个子任务的优先级，O[i][1]表示的是第i个任务的前序最大优先级

def reform_task(task_set):
    task_size = np.shape(task_set)[0]
    task_num = int(task_set[task_size-1][1])#大任务个数
    graph = np.zeros([task_num])
    for i in range(task_num):
        #task_graphtype_num[i][0] = 0
        graph[i] = 1
    flag = 0
    temp = np.zeros([task_size,1])
    darenwu = np.zeros([task_num])
    k = 0
    for i in range(task_size):
        if flag == 0:
            oprnum = opr_num(task_set,i) # 当前i子任务对应的大任务的活动数量
            darenwu[k] = oprnum
            k = k + 1
            graph_type = graph[oprnum-1] # 确定任务对应的图架构[活动数量]，表示第几个类型的图
            AOE = choose_graph(graph_type,oprnum) # 求出AOE
            O = priority(AOE)

        temp[i][0] = O[flag][0]
        flag = flag + 1
        if i != task_size-1:
            if task_set[i+1][2] == 1:
                flag = 0
    task_set = np.hstack((task_set, temp))





    return task_set

if __name__ == '__main__':

    machine = Machine.Machine()
    task = Task.Task()
    task_set = task.load_task(1, 1)

    k = reform_task(task_set)

    print(k)