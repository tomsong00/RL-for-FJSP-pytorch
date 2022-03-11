import numpy as np
from Priority import priority
from Opr_num import opr_num
from Choose_graph import choose_graph
import Machine
import Task
from Order import order
import itertools

#函数返回每一个子任务的优先级
# O[i][0]表示第i个子任务的优先级，O[i][1]表示的是第i个任务的前序最大优先级

def instantiate_task(task_set):
    task_size = np.shape(task_set)[0]
    task_num = int(task_set[task_size-1][1])#大任务个数
    graph = np.zeros([task_num])
    for i in range(task_num):
        #task_graphtype_num[i][0] = 0
        graph[i] = 1
    flag = 0
    temp = np.zeros([task_size,1])
    darenwu = np.zeros([task_num])
    darenwu_graph = []
    k = 0
    for i in range(task_size):
        if flag == 0:
            oprnum = opr_num(task_set,i)  # 当前i子任务对应的大任务的活动数量
            darenwu[k] = oprnum     ################## 大任务的活动数量数组 ##################
            graph_type = graph[oprnum-1]  # 确定任务对应的图架构[活动数量]，表示第几个类型的图
            AOE = choose_graph(graph_type,oprnum)  # 求出AOE
            O = priority(AOE)
            list(itertools.chain.from_iterable(AOE))
            darenwu_graph.append(AOE)  ##################  大任务的活动图数组  ##################
            k = k + 1
        temp[i][0] = O[flag][0]
        flag = flag + 1
        if i != task_size-1:
            if task_set[i+1][2] == 1:
                flag = 0
    #temp1 = task_set[:,2].tolist()
    #temp1=np.array(temp1)
    #temp1 = temp1.transpose()
    b = np.shape(task_set)[0]
    temp1 = np.zeros([b,1])
    for i in range(b):
        temp1[i][0] = task_set[i][2]
    task_set=np.hstack([task_set,temp1])
    #task_set=temp1
    #task_set = np.hstack((task_set, np.reshape(task_set[:,2],(-1,task_set.shape[0]))))

    #对每个大任务的各个子任务按照任务优先级和任务执行拓扑图生成执行顺序
    hang = np.shape(task_set)[0]
    lie = np.shape(task_set)[1]
    task_set_new = np.zeros([hang,lie])
    flag_task_id = 0 # 当前调整的任务序号
    flag_task_id_ini = 0 # 当前调整的大任务的初始位置
    for i in range(task_num):
        task_sequence = []

        task_sequence = order(darenwu_graph[i]) # index = 任务编号; task_sequence[i] = 编号 i 的任务的执行序号
        huodong_num = int(darenwu[i])
        for j in range(int(huodong_num)): # 处理当前第 i 个大任务的huodong_num个活动对应的task行
            s = int(task_sequence[j])
            task_set_new[flag_task_id_ini + s - 1,:] = task_set[flag_task_id,:]
            flag_task_id = flag_task_id + 1
        for p in range(huodong_num):
            task_set_new[p+flag_task_id_ini,2] = p + 1
        flag_task_id_ini = flag_task_id_ini + int(huodong_num)
    return task_set_new

if __name__ == '__main__':

    machine = Machine.Machine()
    task = Task.Task()
    task_set = task.load_task(1, 1)

    k = instantiate_task(task_set)

    print(k)