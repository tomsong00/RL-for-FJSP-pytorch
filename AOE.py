
class Graph:  # basic graph class, using adjacent matrix
    def __init__(self, mat, unconn=0):
        vnum1 = len(mat)  # 行数
        for x in mat:
            if len(x) != vnum1:  # Check square matrix检查是否为方阵
                raise ValueError("Argumentfor 'GraphA' is bad.")
        self.mat = [mat[i][:] for i in range(vnum1)]
        self.unconn = unconn
        self.vnum = vnum1
    def _out_edges(mat, vi, unconn):
        edges = []
        row = mat[vi]
        for i in range(len(row)):
            if row[i] != unconn:
                # print(edges) #都为空
                edges.append((i, row[i]))
                # print(edges)
        return edges
    def vertex_num(self):
        return self.vnum

class GraphA(Graph):

    def __init__(self, mat, unconn=0):
        vnum1 = len(mat)
        for x in mat:
            if len(x) != vnum1:  # Check squarematrix，检查是否为方阵
                raise ValueError("Argumentfor 'GraphA' is bad.")
        self.mat = [Graph._out_edges(mat, i, unconn)
                    for i in range(vnum1)]
        self.vnum = vnum1
        self.unconn = unconn

    def add_vertex(self):  # For new vertex,return an index allocated
        self.mat.append([])
        self.vnum += 1
        return self.vnum

    def add_edge(self, vi, vj, val=1):
        assert 0 <= vi < self.vnum and 0 <= vj < self.vnum
        row = self.mat[vi]
        for i in range(len(row)):
            if row[i][0] == vj:  # replace avalue for mat[vi][vj]
                self.mat[vi][i] = (vj, val)
                return
            if row[i][0] > vj:
                break
        else:
            i += 1  # adjust for the new entryat the end
            self.mat[vi].insert(i, (vj, val))

    def get_edge(self, vi, vj):
        assert 0 <= vi < self.vnum and 0 <= vj < self.vnum
        for i, val in self.mat[vi]:
            if i == vj:
                return val
        return self.unconn

    def out_edges(self, vi):
        assert 0 <= vi < self.vnum;
        return self.mat[vi]

    def toposort(self,graph):
        vnum = graph.vertex_num()
        indegree, toposeq, zerov = [0] * vnum, [], -1
        for vi in range(vnum):
            for v, w in graph.out_edges(vi):
                indegree[v] += 1
        for vi in range(vnum):
            if indegree[vi] == 0:
                indegree[vi] = zerov;
                zerov = vi
        for n in range(vnum):
            if zerov == -1:
                return False  # Thereisno topo-seq
            toposeq.append(zerov)
            vi = zerov
            zerov = indegree[zerov]
            for v, w in graph.out_edges(vi):
                indegree[v] -= 1
                if indegree[v] == 0:
                    indegree[v] = zerov
                    zerov = v
        return toposeq

    def criticalPath(self,graph):
        toposeq = self.toposort(graph)
        if toposeq == False:
            return False  # no topo-sequence, cannotcontinue
        vnum = graph.vertex_num()
        ee, le = [0] * vnum, [inf] * vnum
        crtPath = []
        self.setEventE(vnum, graph, toposeq, ee)
        print(ee)
        self.setEventL(vnum, graph, toposeq, ee[vnum - 1], le)
        print(le)
        for i in range(vnum):
            for j, w in graph.out_edges(i):
                if ee[i] == le[j] - w:  # 列表中是列表，关键活动
                    crtPath.append([i, j, ee[i]])
        return crtPath  # 返回关键活动

    def setEventE(self,vnum, graph, toposeq, ee):
        for k in range(vnum - 1):  # 最后一个顶点不必做
            i = toposeq[k]
            for j, w in graph.out_edges(i):
                if ee[i] + w > ee[j]:  # 事件j需更晚结束
                    ee[j] = ee[i] + w

    def setEventL(self,vnum, graph, toposeq, eelast, le):
        for i in range(vnum):
            le[i] = eelast
        for k in range(vnum - 2, -1, -1):  # 逆拓扑排序，最后顶点不必做
            i = toposeq[k]
            for j, w in graph.out_edges(i):
                if le[j] - w < le[i]:  # 事件i需更早时间
                    le[i] = le[j] - w

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
    inf = float("inf")  # infinity
    g6 = GraphA(gmat6)
    toposeq = g6.toposort(g6)
    print(toposeq)
    gmat7 = [[inf, 6, 4, 5, inf, inf, inf, inf, inf],
             [inf, inf, inf, inf, 1, inf, inf, inf, inf],
             [inf, inf, inf, inf, 1, inf, inf, inf, inf],
             [inf, inf, inf, inf, inf, 2, inf, inf, inf],
             [inf, inf, inf, inf, inf, inf, 9, 7, inf],
             [inf, inf, inf, inf, inf, inf, inf, 4, inf],
             [inf, inf, inf, inf, inf, inf, inf, inf, 2],
             [inf, inf, inf, inf, inf, inf, inf, inf, 4],
             [inf, inf, inf, inf, inf, inf, inf, inf, inf]]
    g7 = GraphA(gmat7, inf)
    cp = g7.criticalPath(g7)
    print(cp)
    pass