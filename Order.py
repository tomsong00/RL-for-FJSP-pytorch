import numpy as np
import random
#函数返回每一个子任务的优先级
# O[i][0]表示第i个子任务的优先级，O[i][1]表示的是第i个任务的前序最大优先级

def order(AOE):
    m = np.shape(AOE)[0] # huodong 数量
    O = np.zeros([m, 2])

    for i in range(m):
        O[i][0] = 1
        z = 0
        for j in range(m):
            if AOE[j][i] == 1:
                if O[j][0] >= z:
                    z = int(O[j][0])
        O[i][1] = z
        if O[i][1] != 0:
            O[i][0] = O[i][0] + O[i][1]
            i = i + 1
        else:
            i = i + 1

    hd_shunxu = np.zeros([m, 1])
    flag = np.zeros([m])
    flag_ready = np.zeros([m])
    shunxu = 1
    for i in range(m):#给i安排下一个活动

        if i == 0:
            flag[i] = 1 # 表示此次选择执行的是编号为1的活动
            hd_shunxu[i] = 1
            shunxu = shunxu + 1
        else:
            daixuan = []
            for j in range(m): # j表示待选活动编号，意味着可以选择执行（找到所有可选后续活动）
                b = 0

                for h in range(m):

                    if AOE[h][j] == 1 and flag[h] == 1 and flag[j] == 0:
                        if flag[h] == 1:
                            b = b + 1
                        else:
                            b = b * 0

                if b != 0:
                    flag_ready[j] = 1
                    daixuan.append(j)
                else:
                    flag_ready[j] = 0

            num = len(daixuan)
            if num != 0:
                rand_num = random.randint(0, num - 1)
            else:
                continue

            xuandinghuodong_num = daixuan[rand_num]
            flag[xuandinghuodong_num] = 1

            hd_shunxu[xuandinghuodong_num] = shunxu
            shunxu = shunxu + 1

    return hd_shunxu # 编号为 i 的活动的实际执行顺序


if __name__ == '__main__':

    gmat6 = [[0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0]]

    k = order(gmat6)

    print(k)