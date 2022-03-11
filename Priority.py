import numpy as np
#函数返回每一个子任务的优先级
# O[i][0]表示第i个子任务的优先级，O[i][1]表示的是第i个任务的前序最大优先级

def priority(AOE):
    m = np.shape(AOE)[0]
    O = np.zeros([m, 2])
    for i in range(m):
        O[i][0] = 1
        z = 0
        for j in range(m):
            if AOE[j][i] == 1:
                if  O[j][0] >= z:
                    z = int(O[j][0])
        O[i][1] = z
        if O[i][1] != 0:
            O[i][0] = O[i][0] + O[i][1]
            i = i + 1
        else:
            i = i + 1
    return O

if __name__ == '__main__':

    gmat6 = [[0, 0, 1, 0, 0, 0, 0, 1, 0],
             [0, 0, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 1, 0, 0]]

    k=priority(gmat6)

    print(k)