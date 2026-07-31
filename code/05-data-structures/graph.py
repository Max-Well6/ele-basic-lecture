"""
图 —— 数据结构讲义配套示例
运行：python graph.py

包含：
1) 邻接矩阵 / 邻接表两种存储
2) BFS（层序 + 最短路径跳数）、DFS（递归 + 栈迭代）
3) 拓扑排序（Kahn 入度法，检测环）
4) 关键路径（AOE 网 ve/vl 推算，找关键活动）
5) Dijkstra 单源最短路（用堆优化）
6) 并查集 + Kruskal 最小生成树
"""

import heapq


# ============================================================
# 一、图的存储
# ============================================================
class Graph:
    """无向/有向图，邻接表存储：{u: [(v, w), ...]}"""

    def __init__(self, directed=False):
        self.adj = {}
        self.directed = directed

    def add_vertex(self, u):
        self.adj.setdefault(u, [])

    def add_edge(self, u, v, w=1):
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].append((v, w))
        if not self.directed:
            self.adj[v].append((u, w))

    def vertices(self):
        return sorted(self.adj.keys())

    def to_matrix(self):
        """转邻接矩阵：稠密图省空间查询快 O(1)，稀疏图浪费 O(V^2)"""
        vs = self.vertices()
        idx = {v: i for i, v in enumerate(vs)}
        n = len(vs)
        m = [[0] * n for _ in range(n)]
        for u in vs:
            for v, w in self.adj[u]:
                m[idx[u]][idx[v]] = w
        return vs, m

    # -------- 遍历 --------
    def bfs(self, start):
        """广度优先：队列，逐层扩散，可求无权图最短跳数"""
        visited = {start}
        queue = [start]
        order, dist = [], {start: 0}
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v, _ in sorted(self.adj[u]):
                if v not in visited:
                    visited.add(v)
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return order, dist

    def dfs_recursive(self, start):
        """深度优先（递归版）：一条路走到黑再回溯"""
        visited, order = set(), []

        def _go(u):
            visited.add(u)
            order.append(u)
            for v, _ in sorted(self.adj[u]):
                if v not in visited:
                    _go(v)
        _go(start)
        return order

    def dfs_iterative(self, start):
        """深度优先（显式栈版）：等价于把递归栈手动管理"""
        visited, order, stack = set(), [], [start]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            order.append(u)
            # 逆序入栈保证访问顺序与递归版一致
            for v, _ in sorted(self.adj[u], reverse=True):
                if v not in visited:
                    stack.append(v)
        return order


# ============================================================
# 二、拓扑排序（Kahn 入度法）
# ============================================================
def topological_sort(graph):
    """有向无环图 DAG 的线性序；若存在环则返回 None"""
    indeg = {u: 0 for u in graph.adj}
    for u in graph.adj:
        for v, _ in graph.adj[u]:
            indeg[v] += 1

    queue = sorted([u for u in indeg if indeg[u] == 0])
    order = []
    print("  初始入度:", dict(sorted(indeg.items())))
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v, _ in graph.adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
        queue.sort()
        print(f"  取出 {u}, 剩余队列 {queue}")
    return order if len(order) == len(indeg) else None


def critical_path(n, edges):
    """AOE 网关键路径。edges: [(起点, 终点, 耗时)]，节点编号 0..n-1
       ve = 事件最早发生时间（正向递推取最大）
       vl = 事件最迟发生时间（逆向递推取最小）
       活动余量 = 最迟开始 - 最早开始，为 0 者即关键活动"""
    succ = {u: [] for u in range(n)}
    indeg = {u: 0 for u in range(n)}
    for u, v, w in edges:
        succ[u].append((v, w))
        indeg[v] += 1

    order, queue = [], [u for u in range(n) if indeg[u] == 0]
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v, _ in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if len(order) != n:
        return None                             # 有环，不是 AOE 网

    ve = {u: 0 for u in range(n)}
    for u in order:
        for v, w in succ[u]:
            ve[v] = max(ve[v], ve[u] + w)
    total = max(ve.values())

    vl = {u: total for u in range(n)}
    for u in reversed(order):
        for v, w in succ[u]:
            vl[u] = min(vl[u], vl[v] - w)

    keys = [(u, v, w) for u, v, w in edges if ve[u] == vl[v] - w]
    return total, keys, ve, vl


# ============================================================
# 三、Dijkstra（非负权单源最短路，堆优化 O((V+E)logV)）
# ============================================================
def dijkstra(graph, src):
    dist = {u: float("inf") for u in graph.adj}
    dist[src] = 0
    prev = {}
    pq = [(0, src)]
    done = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        for v, w in graph.adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))
    return dist, prev


def build_path(prev, src, dst):
    path, cur = [dst], dst
    while cur != src:
        if cur not in prev:
            return None
        cur = prev[cur]
        path.append(cur)
    return list(reversed(path))


# ============================================================
# 四、并查集 + Kruskal 最小生成树
# ============================================================
class UnionFind:
    """并查集：路径压缩 + 按秩合并，均摊接近 O(1)"""

    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # 路径压缩
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False            # 已在同一集合，加边会成环
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def kruskal(graph):
    """按边权从小到大贪心选边，用并查集判环"""
    edges = set()
    for u in graph.adj:
        for v, w in graph.adj[u]:
            edges.add((w, min(u, v), max(u, v)))
    uf = UnionFind(graph.vertices())
    mst, total = [], 0
    for w, u, v in sorted(edges):
        if uf.union(u, v):
            mst.append((u, v, w))
            total += w
    return mst, total


if __name__ == "__main__":
    print("===== 无向图遍历 =====")
    g = Graph()
    for u, v in [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"),
                 ("D", "E"), ("E", "F")]:
        g.add_edge(u, v)

    vs, mat = g.to_matrix()
    print("顶点:", vs)
    print("邻接矩阵:")
    for name, row in zip(vs, mat):
        print(f"  {name}: {row}")

    order, dist = g.bfs("A")
    print("BFS 顺序:", order)
    print("A 到各点跳数:", dist)
    print("DFS 递归:", g.dfs_recursive("A"))
    print("DFS 迭代:", g.dfs_iterative("A"))

    print("\n===== 拓扑排序（课程先修关系）=====")
    dag = Graph(directed=True)
    for u, v in [("程序设计", "数据结构"), ("离散数学", "数据结构"),
                 ("数据结构", "算法分析"), ("数据结构", "数据库"),
                 ("算法分析", "机器学习"), ("数据库", "机器学习")]:
        dag.add_edge(u, v)
    topo = topological_sort(dag)
    print("可行修课顺序:", " -> ".join(topo) if topo else "存在环，无法排序")

    print("\n===== 关键路径（AOE 网，6 个事件）=====")
    total, keys, ve, vl = critical_path(
        6, [(0, 1, 3), (0, 2, 2), (1, 3, 2), (2, 3, 4),
            (2, 4, 3), (3, 5, 2), (4, 5, 1)])
    print("  最短工期:", total)
    for i in range(6):
        print(f"  事件{i}: ve={ve[i]}, vl={vl[i]}, 余量={vl[i]-ve[i]}")
    print("  关键活动:", " ".join(f"{u}->{v}(耗时{w})" for u, v, w in keys))

    print("\n===== Dijkstra 最短路 =====")
    wg = Graph()
    for u, v, w in [("A", "B", 4), ("A", "C", 2), ("B", "C", 1),
                    ("B", "D", 5), ("C", "D", 8), ("C", "E", 10),
                    ("D", "E", 2), ("D", "F", 6), ("E", "F", 3)]:
        wg.add_edge(u, v, w)
    d, prev = dijkstra(wg, "A")
    for k in sorted(d):
        print(f"  A -> {k}: 距离 {d[k]}, 路径 {build_path(prev, 'A', k) if k != 'A' else ['A']}")

    print("\n===== Kruskal 最小生成树 =====")
    mst, total = kruskal(wg)
    for u, v, w in mst:
        print(f"  选边 {u}-{v} 权重 {w}")
    print("MST 总权重:", total)
