import random
import numpy as np
class Operators(object):
    def __init__(self):
        self
    #交叉
    def crossover(self,population,task_set,idx,length):
        task_size = np.shape(task_set)[0]
        r1=np.random.randint(1, (task_size + 1)-length+1)
        #rand1=np.array(rand1)
        rand1=r1
        r2 = np.random.randint(1, (task_size + 1)-length+1)
        rand2=r2
        #rand2 = np.array(rand2)
        while abs(rand2-rand1)<length:
            rand1 = np.random.randint(1, (task_size + 1)-length+1)
            rand2 = np.random.randint(1, (task_size + 1)-length+1)
        #print(population[idx])
        temp_seq1=population[idx,rand1-1:(rand1-1+length)]
        temp_seq2=population[idx, rand2-1:(rand2-1 + length)]
        #array会同步更新，需要转list
        temp_seq1 = temp_seq1.tolist()
        temp_seq2=temp_seq2.tolist()
        population[idx][rand1-1:(rand1-1+length)]=np.array(temp_seq2)
        population[idx][rand2-1:(rand2-1+length)]=np.array(temp_seq1)
        opt_population=population
        #print(population[idx])
        return opt_population

    def mutation(self,population,task_set,idx):
        task_size = np.shape(task_set)[0]
        r1=np.random.randint(1, task_size + 1)
        #rand1=np.array(rand1)
        rand1=r1
        r2 = np.random.randint(1, (task_size + 1))
        rand2=r2
        #rand2 = np.array(rand2)
        while rand1==rand2:
            rand1 = np.random.randint(1, (task_size + 1))
            rand2 = np.random.randint(1, (task_size + 1))
        temp_seq1=population[idx,rand1-1]
        temp_seq2=population[idx, rand2-1]
        #array会同步更新，需要转list
        temp_seq1 = temp_seq1.tolist()
        temp_seq2=temp_seq2.tolist()
        population[idx,rand1-1]=temp_seq2
        population[idx,rand2-1]=temp_seq1
        opt_population=population
        #print(population[idx])
        return opt_population
