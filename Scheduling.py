import numpy as np
from Instantiate_task import instantiate_task

class Scheudling:
    def __init__(self):
        self

    # 给设备安排任务
    def arrange_task_to_machine(self, population, task_set, machine_set,information_network):
        population_size = np.shape(population)[0]
        task_size = np.shape(task_set)[0]
        task_attri_size = np.shape(task_set)[1]
        # 存储结果
        g = int(np.shape(task_set)[1]) + 2
        pop_result = np.zeros([population_size, int(task_size), g]) #“+2”指的是加了machine ID和原有活动编号
        temp_task = np.zeros([task_size,task_attri_size])
        for i in range(task_size):
            for j in range(task_attri_size):
                temp_task[i][j] = task_set[i][j]
        # 依次读取种群每一行
        for i in range(population_size):
            #实例化新的task_set
            task_set = instantiate_task(temp_task)
            task_num = np.shape(task_set)[0]
            for g in range(task_num):
                task_set[g][0] = g + 1
            # 需要分割时间窗，所以增加一个新的机器表，在此操作
            arrange_number = 0
            temp_machine_set = np.array(machine_set)
            machine_size = np.shape(temp_machine_set)[0]
            # 初始化一个记录前序任务的表,第一个值记录前序编号
            indi_result = np.zeros([1, int(np.shape(task_set)[1] + 1)])
            processor = np.zeros([int(task_set[task_size - 1][1]), 3]) # 任务前序任务编号和完成时间和完成机器编号，行=大任务数量
            # 0:id  1:所属大任务  2:内部编号  3:允许执行的机器类型编号  4:准备时间  5:加工时间 6:机器编号
            # 依次读取每行中的每一列
            for j in range(task_size): #第 j 个任务
                task_id = np.int(population[i][j])  #任务编号
                # 需要根据前序完成转到对应后序任务
                job_id = task_set[int(task_id - 1)][1]   # job_id = 任务所属大任务编号
                if np.int(processor[int(job_id - 1)][0]) == 0: #判断当前大任务是否执行了第一个子任务
                    # 找到每一个第一个子任务
                    col_job = task_set[:, 1]
                    subtask = np.where(col_job == job_id) #where返回的是跟当前大任务一致的小任务编号
                    subtask = np.array(subtask)
                    task_id = np.int(task_set[subtask[0][0]][0]) #task_id是从1开始的
                if np.int(processor[int(job_id - 1)][0]) != 0:
                    task_id = np.int(processor[int(job_id - 1)][0]+1) #task_id是从1开始的，因此+1
                # 遍历机器表
                for k in range(machine_size):
                    temp = np.int(processor[int(job_id - 1)][2]) # 任务前序任务完成机器编号
                    temp1 = np.int(temp_machine_set[k][0]) # 第k个机器实际实物编号
                    # 任务要求类型与机器类型相同
                    if np.int(task_set[int(task_id - 1)][3]) == np.int(temp_machine_set[k][1]) and \
                            np.int(information_network[temp-1][temp1-1] == 1):
                        # 判断时间窗大小
                        duration = np.int(
                            temp_machine_set[k][3] - max(temp_machine_set[k][2], processor[int(job_id - 1)][1]))
                        if np.int(task_set[int(task_id - 1)][4] + task_set[int(task_id - 1)][5]) <= duration:
                            # 根据前序任务分为两种情况
                            # 直接安排
                            if (np.int(processor[int(job_id - 1)][1]) <= np.int(temp_machine_set[k][2]) and
                                    np.int(processor[int(job_id - 1)][1] + task_set[int(task_id - 1)][4] +
                                           task_set[int(task_id - 1)][5]) <= np.int(temp_machine_set[k][3])):
                                if arrange_number == 0:
                                    # 创建一个array
                                    temp_indi_result = np.hstack(
                                        (task_set[int(task_id - 1)][:], temp_machine_set[k][0]))
                                    end_time = temp_machine_set[k][2] + temp_indi_result[4] + temp_indi_result[5]
                                    temp_indi_result[5] = end_time
                                    temp_indi_result[4] = temp_machine_set[k][2]
                                    indi_result[0][:] = temp_indi_result
                                    # 添加前序记录表
                                    processor[int(job_id - 1)][0] = temp_indi_result[0]
                                    processor[int(job_id - 1)][1] = temp_indi_result[5]
                                    processor[int(job_id - 1)][2] = temp_indi_result[6]
                                    # 修改机器表
                                    temp_machine_set[k][2] = temp_indi_result[5]
                                else:
                                    # 直接在array中继续添加
                                    temp_indi_result = np.hstack(
                                        (task_set[int(task_id - 1), :], temp_machine_set[k][0]))
                                    end_time = temp_machine_set[k][2] + temp_indi_result[4] + temp_indi_result[5]
                                    temp_indi_result[5] = end_time
                                    temp_indi_result[4] = temp_machine_set[k][2]
                                    indi_result = np.vstack((indi_result, temp_indi_result))
                                    # 添加前序记录表
                                    processor[int(job_id - 1)][0] = temp_indi_result[0]
                                    processor[int(job_id - 1)][1] = temp_indi_result[5]
                                    processor[int(job_id - 1)][2] = temp_indi_result[6]
                                    # 修改机器表
                                    temp_machine_set[k][2] = temp_indi_result[5]
                                arrange_number = arrange_number + 1
                                break
                            # 切割时间窗
                            if [np.int(processor[int(job_id - 1)][1]) > np.int(temp_machine_set[k][2]) and (np.int(
                                    processor[int(job_id - 1)][1] + task_set[int(task_id - 1)][4] +
                                    task_set[int(task_id - 1)][5]) <= np.int(temp_machine_set[k][3])) and np.int(
                                    task_set[int(task_id - 1)][4] + task_set[int(task_id - 1)][5]) <= np.int(
                                    temp_machine_set[k][3] - processor[int(job_id - 1)][1])]:
                                if arrange_number == 0:
                                    # 创建一个array
                                    #最后一列表示安排的机器编号
                                    temp_indi_result = np.hstack(
                                        (task_set[int(task_id - 1), :], temp_machine_set[k][0]))
                                    end_time = processor[int(job_id - 1)][1] + task_set[int(task_id - 1)][4] + \
                                               task_set[int(task_id - 1)][5]
                                    temp_indi_result[5] = end_time
                                    temp_indi_result[4] = processor[int(job_id - 1)][1]
                                    indi_result[0][:] = temp_indi_result
                                    # 添加前序记录表
                                    processor[int(job_id - 1)][0] = temp_indi_result[0]
                                    processor[int(job_id - 1)][1] = temp_indi_result[5]
                                    processor[int(job_id - 1)][2] = temp_indi_result[6]
                                    # 添加新的机器
                                    temp_machine_set = np.vstack((temp_machine_set, temp_machine_set[k, :]))
                                    machine_size = np.shape(temp_machine_set)[0]
                                    temp_machine_set[k][3] = temp_indi_result[4]
                                    #  print(temp_indi_result[4]); print(temp_indi_result[5])
                                    temp_machine_set[machine_size - 1][2] = temp_indi_result[5]
                                else:
                                    # 直接在array中继续添加
                                    temp_indi_result = np.hstack(
                                        (task_set[int(task_id - 1), :], temp_machine_set[k][0]))
                                    end_time = processor[int(job_id - 1)][1] + task_set[int(task_id - 1)][4] + \
                                               task_set[int(task_id - 1)][5]
                                    temp_indi_result[5] = end_time
                                    temp_indi_result[4] = processor[int(job_id - 1)][1]
                                    indi_result = np.vstack((indi_result, temp_indi_result))
                                    # 添加前序记录表
                                    processor[int(job_id - 1)][0] = temp_indi_result[0]
                                    processor[int(job_id - 1)][1] = temp_indi_result[5]
                                    processor[int(job_id - 1)][2] = temp_indi_result[6]
                                    # 添加新的机器
                                    temp_machine_set = np.vstack((temp_machine_set, temp_machine_set[k, :]))
                                    machine_size = np.shape(temp_machine_set)[0]
                                    temp_machine_set[k][3] = temp_indi_result[4]
                                    temp_machine_set[machine_size - 1][2] = temp_indi_result[5]
                                #  print(temp_indi_result[4]);print(temp_indi_result[5])
                                arrange_number = arrange_number + 1
                                break
            pop_result[i][0:np.shape(indi_result)[0]] = indi_result
        return pop_result
    #0-id	1-所属大任务  2-内部编号  3-允许执行的机器类型编号	4-开始时间 5-结束时间 6-机器编号
    # 计算适应度函数值
    def fitness_calculation(self, scehduling_result, populaiton):
        population_size = np.shape(populaiton)[0]
        fitness = np.zeros(population_size)
        for i in range(population_size):
            comlete_time = scehduling_result[i, :, 5]
            fitness[i] = max(comlete_time)
        return fitness

    def arrange_task_to_machine_RL(self, population, task_set, machine_set,information_network,processor):
        population_size = np.shape(population)[0]
        task_size = np.shape(task_set)[0]
        task_attri_size = np.shape(task_set)[1]
        # 存储结果
        g = int(np.shape(task_set)[1]) + 2
        pop_result = np.zeros([population_size, int(task_size), g]) #“+2”指的是加了machine ID和原有活动编号
        temp_task = np.zeros([task_size,task_attri_size])
        for i in range(task_size):
            for j in range(task_attri_size):
                temp_task[i][j] = task_set[i][j]
        # 依次读取种群每一行
        for i in range(population_size):
            #实例化新的task_set
            task_set = instantiate_task(temp_task)
            task_num = np.shape(task_set)[0]
            for g in range(task_num):
                task_set[g][0] = g + 1
            # 需要分割时间窗，所以增加一个新的机器表，在此操作
            arrange_number = 0
            temp_machine_set = np.array(machine_set)
            machine_size = np.shape(temp_machine_set)[0]
            # 初始化一个记录前序任务的表,第一个值记录前序编号
            indi_result = np.zeros([1, int(np.shape(task_set)[1] + 1)])
            #processor = np.zeros([int(task_set[task_size - 1][1]), 3]) # 任务前序任务编号和完成时间和完成机器编号，行=大任务数量
            # 0:id  1:所属大任务  2:内部编号  3:允许执行的机器类型编号  4:准备时间  5:加工时间 6:机器编号
            # 依次读取每行中的每一列
            for j in range(np.shape(population)[1]): #第 j 个任务
                task_id = np.int(population[i][j])  #任务编号
                # 需要根据前序完成转到对应后序任务
                job_id = task_set[int(task_id - 1)][1]   # job_id = 任务所属大任务编号
                if np.int(processor[int(job_id - 1)][0]) == 0: #判断当前大任务是否执行了第一个子任务
                    # 找到每一个第一个子任务
                    col_job = task_set[:, 1]
                    subtask = np.where(col_job == job_id) #where返回的是跟当前大任务一致的小任务编号
                    subtask = np.array(subtask)
                    task_id = np.int(task_set[subtask[0][0]][0]) #task_id是从1开始的
                if np.int(processor[int(job_id - 1)][0]) != 0:
                    task_id = np.int(processor[int(job_id - 1)][0]+1) #task_id是从1开始的，因此+1
                # 遍历机器表
                for k in range(machine_size):
                    temp = np.int(processor[int(job_id - 1)][2]) # 任务前序任务完成机器编号
                    temp1 = np.int(temp_machine_set[k][0]) # 第k个机器实际实物编号
                    # 任务要求类型与机器类型相同
                    if np.int(task_set[int(task_id - 1)][3]) == np.int(temp_machine_set[k][1]) and \
                            np.int(information_network[temp-1][temp1-1] == 1):
                        # 判断时间窗大小
                        duration = np.int(
                            temp_machine_set[k][3] - max(temp_machine_set[k][2], processor[int(job_id - 1)][1]))
                        if np.int(task_set[int(task_id - 1)][4] + task_set[int(task_id - 1)][5]) <= duration:
                            # 根据前序任务分为两种情况
                            # 直接安排
                            if (np.int(processor[int(job_id - 1)][1]) <= np.int(temp_machine_set[k][2]) and
                                    np.int(processor[int(job_id - 1)][1] + task_set[int(task_id - 1)][4] +
                                           task_set[int(task_id - 1)][5]) <= np.int(temp_machine_set[k][3])):
                                if arrange_number == 0:
                                    # 创建一个array
                                    temp_indi_result = np.hstack(
                                        (task_set[int(task_id - 1)][:], temp_machine_set[k][0]))
                                    end_time = temp_machine_set[k][2] + temp_indi_result[4] + temp_indi_result[5]
                                    temp_indi_result[5] = end_time
                                    temp_indi_result[4] = temp_machine_set[k][2]
                                    indi_result[0][:] = temp_indi_result
                                    # 添加前序记录表
                                    processor[int(job_id - 1)][0] = temp_indi_result[0]
                                    processor[int(job_id - 1)][1] = temp_indi_result[5]
                                    processor[int(job_id - 1)][2] = temp_indi_result[6]
                                    # 修改机器表
                                    temp_machine_set[k][2] = temp_indi_result[5]
                                else:
                                    # 直接在array中继续添加
                                    temp_indi_result = np.hstack(
                                        (task_set[int(task_id - 1), :], temp_machine_set[k][0]))
                                    end_time = temp_machine_set[k][2] + temp_indi_result[4] + temp_indi_result[5]
                                    temp_indi_result[5] = end_time
                                    temp_indi_result[4] = temp_machine_set[k][2]
                                    indi_result = np.vstack((indi_result, temp_indi_result))
                                    # 添加前序记录表
                                    processor[int(job_id - 1)][0] = temp_indi_result[0]
                                    processor[int(job_id - 1)][1] = temp_indi_result[5]
                                    processor[int(job_id - 1)][2] = temp_indi_result[6]
                                    # 修改机器表
                                    temp_machine_set[k][2] = temp_indi_result[5]
                                arrange_number = arrange_number + 1
                                break
                            # 切割时间窗
                            if [np.int(processor[int(job_id - 1)][1]) > np.int(temp_machine_set[k][2]) and (np.int(
                                    processor[int(job_id - 1)][1] + task_set[int(task_id - 1)][4] +
                                    task_set[int(task_id - 1)][5]) <= np.int(temp_machine_set[k][3])) and np.int(
                                    task_set[int(task_id - 1)][4] + task_set[int(task_id - 1)][5]) <= np.int(
                                    temp_machine_set[k][3] - processor[int(job_id - 1)][1])]:
                                if arrange_number == 0:
                                    # 创建一个array
                                    #最后一列表示安排的机器编号
                                    temp_indi_result = np.hstack(
                                        (task_set[int(task_id - 1), :], temp_machine_set[k][0]))
                                    end_time = processor[int(job_id - 1)][1] + task_set[int(task_id - 1)][4] + \
                                               task_set[int(task_id - 1)][5]
                                    temp_indi_result[5] = end_time
                                    temp_indi_result[4] = processor[int(job_id - 1)][1]
                                    indi_result[0][:] = temp_indi_result
                                    # 添加前序记录表
                                    processor[int(job_id - 1)][0] = temp_indi_result[0]
                                    processor[int(job_id - 1)][1] = temp_indi_result[5]
                                    processor[int(job_id - 1)][2] = temp_indi_result[6]
                                    # 添加新的机器
                                    temp_machine_set = np.vstack((temp_machine_set, temp_machine_set[k, :]))
                                    machine_size = np.shape(temp_machine_set)[0]
                                    temp_machine_set[k][3] = temp_indi_result[4]
                                    #  print(temp_indi_result[4]); print(temp_indi_result[5])
                                    temp_machine_set[machine_size - 1][2] = temp_indi_result[5]
                                else:
                                    # 直接在array中继续添加
                                    temp_indi_result = np.hstack(
                                        (task_set[int(task_id - 1), :], temp_machine_set[k][0]))
                                    end_time = processor[int(job_id - 1)][1] + task_set[int(task_id - 1)][4] + \
                                               task_set[int(task_id - 1)][5]
                                    temp_indi_result[5] = end_time
                                    temp_indi_result[4] = processor[int(job_id - 1)][1]
                                    indi_result = np.vstack((indi_result, temp_indi_result))
                                    # 添加前序记录表
                                    processor[int(job_id - 1)][0] = temp_indi_result[0]
                                    processor[int(job_id - 1)][1] = temp_indi_result[5]
                                    processor[int(job_id - 1)][2] = temp_indi_result[6]
                                    # 添加新的机器
                                    temp_machine_set = np.vstack((temp_machine_set, temp_machine_set[k, :]))
                                    machine_size = np.shape(temp_machine_set)[0]
                                    temp_machine_set[k][3] = temp_indi_result[4]
                                    temp_machine_set[machine_size - 1][2] = temp_indi_result[5]
                                #  print(temp_indi_result[4]);print(temp_indi_result[5])
                                arrange_number = arrange_number + 1
                                break
            pop_result[i][0:np.shape(indi_result)[0]] = indi_result
        return pop_result,temp_machine_set,processor

    def fitness_calculation_RL(self, scehduling_result, populaiton):
        population_size = np.shape(populaiton)[0]
        fitness = np.zeros(population_size)
        for i in range(population_size):
            comlete_time = scehduling_result[i, :, 5]
            fitness[i] = max(comlete_time)
        return fitness



