import numpy as np

def state_transform(s,s_,a):
    shuxing = np.shape(s)[1]
    s_[a][shuxing - 1] = 1
    return s_