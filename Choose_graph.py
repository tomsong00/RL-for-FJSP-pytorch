import numpy as np


def choose_graph(graph_type,oprnum):
    AOE=[]
    if oprnum == 4:
        if graph_type == 1:
            AOE = [[0, 1, 1, 0],
                   [0, 0, 0, 1],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]]
    if oprnum == 5:
        if graph_type == 1:
            AOE = [[0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0]]
        elif graph_type == 2:
            AOE = [[0, 1, 1, 0, 0],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0]]
        else:
            AOE = [[0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0]]
    if oprnum == 6:
        if graph_type == 1:
            AOE = [[0, 1, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0]]
    if oprnum == 7:
        if graph_type == 1:
            AOE = [[0, 1, 0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0]]
    if oprnum == 9:
        if graph_type == 1:
            AOE = [[0, 1, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 1, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    return AOE