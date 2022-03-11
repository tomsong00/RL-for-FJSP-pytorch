import numpy as np


def opr_num(task_set,i):
    m = np.shape(task_set)[0]
    h = 0
    for k in range(i,m):
        h = h + 1
        if k+1 != m:
            if task_set[k+1][2] == 1:
                break
    return h