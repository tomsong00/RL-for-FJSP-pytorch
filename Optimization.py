import numpy as np
import random
import Scheduling
import Selection
import Operators
import Nature_DQN
class Optimization(object):
    def __init__(self):
        self

    # 初始化种群
    def initialization(self, population_size, task_set):
        task_size = np.shape(task_set)[0]
        self.population = np.empty([population_size, task_size],dtype='int64')
        for row in range(population_size):
            # 上界无法取到，所以需要+1
            rand_list = random.sample(range(1, task_size + 1), task_size)
            self.population[row, :] = np.array(rand_list,dtype='int64')
            #print(row)
        return self.population

    #迭代搜索
    def search_by_iteration(self,population,max_generation,task_set,machine_set,alpha,beta,length,gobal_best_fitness,information_network):
        # 初始化网络


        scheduling=Scheduling.Scheudling()
        selection=Selection.Selection()
        operators=Operators.Operators()

        for gen in range(1,max_generation+1):
            ##安排任务
            #gobal_best_individual = np.zeros([np.shape(task_set)[0], np.shape(task_set)[1] + 1])

            scheduling_result=scheduling.arrange_task_to_machine(population,task_set,machine_set,information_network)

            fitness = scheduling.fitness_calculation(scheduling_result, population)


            if min(fitness)<gobal_best_fitness:
                gobal_best_fitness=min(fitness)
                idx=np.where(fitness==min(fitness))
                temp_gobal_best_individual=scheduling_result[idx[0]]
               # del gobal_best_individual
                gobal_best_individual=temp_gobal_best_individual
                gobal_best_individual = np.reshape(gobal_best_individual, (-1,np.shape(task_set)[1] + 2)).tolist()
            #print("第{}代适应度值为{}".format(gen, gobal_best_fitness))
            probability=selection.generate_probability(fitness,population)
            #遗传进化
            for i in range(np.shape(population)[0]):
                if np.random.rand()<alpha:
                    # 选择操作
                    idx=selection.routewheel(probability,population)
                    population=operators.crossover(population,task_set,idx,length)
                if np.random.rand()<beta:
                    #交叉
                    idx=selection.routewheel(probability,population)
                    population=operators.mutation(population,task_set,idx)

        return gobal_best_fitness,gobal_best_individual


