import GA_main
import numpy as np
import pandas as pd
class Cluster(object):
    def __init__(self):
        self.result_return=GA_main.GA_main()

    def distance_clusters(self,x,y):
        WI = 0.5
        WE = 0.5
        ntask = np.shape(x)[0] - 1
        inter = x[0] + y[0]
        exter = 0
        for i in range(ntask):
            exter = exter + min(x[i+1],y[i+1])
        distance = WI * inter + WE * exter
        return distance

    def clustering(self, w,Q,QQ,nmachine,typenum_task,num_group):
        #Q表示的是cluster的组成，例子中为四个cluster
        #QQ表示的是cluster的资源个数和对应的执行任务的组成
        #w：任务-资源矩阵，nmachine：机器数量，typenum_task：大任务数量，num_group：cluster个数
        numgroup = nmachine
        m = 0
        n = 0
        while numgroup != num_group:
            distance = np.zeros([numgroup, numgroup])
            min = float('inf')
            for i in range(numgroup):
                for j in range(numgroup):
                    distance[i][j] = self.distance_clusters(QQ[i,:], QQ[j,:])
                    if distance[i][j] < min:
                        min = distance[i][j]
                        m = i
                        n = j
            QQ[m][0] = QQ[m][0]  + QQ[n][0]
            for x in range(typenum_task):
                QQ[m][x] = max(w[x][m], w[x, n])
            for i in range(nmachine):
                Q[m][i] = max(Q[m][i], Q[n][i])
            for g in range(n+1,numgroup):
                for h in range(typenum_task):
                    temp=QQ[g][h].tolist()
                    QQ[g-1][h] = np.array(temp)
            for g in range(n+1,numgroup):
                for h in range(nmachine):
                    temp = Q[g][h].tolist()
                    Q[g-1][h] = np.array(temp)
            QQ[numgroup-1][0] = 0
            for x in range(typenum_task):
                QQ[numgroup-1][x + 1] = 0
            for x in range(nmachine):
                Q[numgroup-1][x] = 0
            numgroup = numgroup - 1
        return Q,QQ

    def fenqun(self):
        cluster = Cluster()
        #cluster.result_return
        #RR = cluster.result_return.gobal_best_individual  # 最优调度结果
        table_name = "./data/machine-{}.xlsx".format(1)
        sheet_name = "Sheet{}".format(1)
        data = pd.ExcelFile(table_name)
        machine_set = data.parse(sheet_name, index_col="机器编号")
        machine_set = np.array(machine_set, dtype='int64')
        machine_num = np.shape(machine_set)[0]
        information_network = np.ones([machine_num, machine_num])
        for i in range(5, 13):
            information_network[0][i] = 0
            information_network[1][i] = 0
            information_network[2][i] = 0
        [RR,machine_set,task_set] = cluster.result_return.GA(information_network)
        # RR=np.array(RR,dtype='int64')
        nmachine = np.shape(machine_set)[0]  # 机器个数
        ntask = np.shape(task_set)[0]  # 任务个数
        R_R = np.zeros([nmachine, nmachine])  # 资源信息连边矩阵
        task = task_set
        typenum_task = max(task[:, 1])  # 大任务的数量
        Typenum_task = np.zeros([1, typenum_task])  # 临时变量，记录当前大任务的前序资源编号
        for i in range(ntask):
            temp = np.int(RR[i][1]) - 1
            temp1 = np.int(RR[i][6]) - 1
            if Typenum_task[0][temp] != 0:
                R_R[temp1][np.int(Typenum_task[0][temp])] = 1
                R_R[np.int(Typenum_task[0][[temp]])][temp1] = 1
            Typenum_task[0][temp] = temp1
        # 依据求解出的资源连接矩阵进而求出分群结果
        num_group = 4  # group的个数
        #   分群的结果是使得权重负载值最小的分群方案，即资源在各个群的分布
        # bestrecords % % 任务执行方案表
        w = np.zeros([typenum_task, nmachine])  # w(大任务，资源)
        for i in range(ntask):
            temp2 = np.int(RR[i - 1][1]) - 1
            temp3 = np.int(RR[i - 1][6]) - 1
            w[temp2][temp3] = 1
        Q = np.zeros([nmachine, nmachine])
        QQ = np.zeros([nmachine, typenum_task + 1])
        for i in range(nmachine):  # 资源的维度
            for j in range(typenum_task):  # 大任务的维度
                QQ[i][j + 1] = w[j][i]  # 资源i被指派给大任务j
                QQ[i][0] = 1
                Q[i][i] = 1
        cluster.clustering(w, Q, QQ, nmachine, typenum_task, num_group)
        return QQ,Q

if __name__ == '__main__':
    cluster = Cluster()
    QQ,Q = cluster.fenqun()
    print('f')


    
    
    











