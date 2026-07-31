"""
图算法合集：最短路与最小生成树
==============================
包含 Dijkstra（逐步松弛过程打印）、Bellman-Ford（含负环检测）、Floyd-Warshall、
Prim 与 Kruskal 五个算法的教学实现。

运行方式：
    python dijkstra.py
"""

import heapq

INF = float("inf")


# ======================================================================
# 示例图（无向带权图，用邻接表表示），6 个结点 A~F：
#
#          4          5
#      A ------ B --------- D
#      |       /|          /|
#     2|     1/ |8       2/ |6
#      |     /  |        /  |
#      C ---+   +---E---+   |
#           10        3     |
#                     +-----F
#
# 边集：A-B(4) A-C(2) B-C(1) B-D(5) C-D(8) C-E(10) D-E(2) D-F(6) E-F(3)
# ======================================================================
GRAPH = {
    "A": [("B", 4), ("C", 2)],
    "B": [("A", 4), ("C", 1), ("D", 5)],
    "C": [("A", 2), ("B", 1), ("D", 8), ("E", 10)],
    "D": [("B", 5), ("C", 8), ("E", 2), ("F", 6)],
    "E": [("C", 10), ("D", 2), ("F", 3)],
    "F": [("D", 6), ("E", 3)],
}


# ======================================================================
# 1. Dijkstra —— 单源最短路，要求边权非负
#    思想：贪心地每次取出"当前已知距离最小且未确定"的结点，确定它的最短距离，
#          再用它去松弛（relax）所有邻居。
#    复杂度：二叉堆实现 O((V + E) log V)
#    正确性依赖：边权非负 => 已出堆的结点距离不可能再被改小
# ======================================================================
def dijkstra(graph, source, verbose=True):
    dist = {v: INF for v in graph}
    prev = {v: None for v in graph}
    dist[source] = 0
    visited = set()
    pq = [(0, source)]          # 小顶堆，元素为 (当前距离, 结点)
    step = 0

    if verbose:
        print(f"  源点 = {source}")
        print("  " + "-" * 62)
        print(f"  {'步':<3}{'出堆结点':<10}{'确定距离':<10}松弛情况")
        print("  " + "-" * 62)

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:        # 惰性删除：堆里可能残留旧的、更大的距离记录
            continue
        visited.add(u)
        step += 1
        relaxed = []

        for v, w in graph[u]:
            if v in visited:
                continue
            nd = d + w
            if nd < dist[v]:                       # 松弛成功：发现更短的路
                relaxed.append(f"{v}: {fmt(dist[v])} -> {nd}")
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

        if verbose:
            info = ", ".join(relaxed) if relaxed else "（无更新）"
            print(f"  {step:<3}{u:<10}{d:<10}{info}")

    if verbose:
        print("  " + "-" * 62)
        print("  最终距离表：", {v: fmt(dist[v]) for v in sorted(dist)})
    return dist, prev


def fmt(x):
    """把 inf 显示成 ∞ 的 ASCII 替代，避免终端编码问题。"""
    return "INF" if x == INF else x


def build_path(prev, source, target):
    """根据前驱数组还原从 source 到 target 的完整路径。"""
    path, cur = [], target
    while cur is not None:
        path.append(cur)
        if cur == source:
            break
        cur = prev[cur]
    path.reverse()
    return path if path and path[0] == source else None


# ======================================================================
# 2. Bellman-Ford —— 允许负权边，可检测负环
#    思想：对所有边做 V-1 轮松弛。第 k 轮结束后，"最多经过 k 条边"的最短路已正确。
#    复杂度：O(V * E)
#    第 V 轮若仍能松弛，说明存在负权环（最短路无定义）。
# ======================================================================
def bellman_ford(vertices, edges, source, verbose=True):
    dist = {v: INF for v in vertices}
    dist[source] = 0

    for round_no in range(1, len(vertices)):
        changed = False
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if verbose:
            print(f"  第 {round_no} 轮后：", {v: fmt(dist[v]) for v in vertices})
        if not changed:                # 提前收敛，后面的轮次不会再变化
            if verbose:
                print(f"  第 {round_no} 轮无更新，提前结束")
            break

    # 再跑一轮检测负环
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            return dist, True          # 存在负权环
    return dist, False


# ======================================================================
# 3. Floyd-Warshall —— 全源最短路
#    状态定义：d[k][i][j] = 只允许经过前 k 个结点作为中转时 i->j 的最短距离
#    转移方程：d[k][i][j] = min(d[k-1][i][j], d[k-1][i][k] + d[k-1][k][j])
#    边界条件：d[0][i][j] = 边权（无边则为 INF，自环为 0）
#    实现时 k 这一维可以滚动掉，所以只用二维数组。复杂度 O(V^3)。
# ======================================================================
def floyd_warshall(vertices, edges):
    idx = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
    for u, v, w in edges:
        d[idx[u]][idx[v]] = min(d[idx[u]][idx[v]], w)

    for k in range(n):                 # 中转点必须放在最外层循环！
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d, idx


# ======================================================================
# 4. Prim —— 最小生成树，从一个结点出发不断吸纳最近的邻居
#    复杂度：堆实现 O(E log V)。适合稠密图。
# ======================================================================
def prim(graph, start, verbose=True):
    visited = {start}
    pq = [(w, start, v) for v, w in graph[start]]
    heapq.heapify(pq)
    mst, total = [], 0

    while pq and len(visited) < len(graph):
        w, u, v = heapq.heappop(pq)
        if v in visited:               # v 已在树中，这条边会成环，丢弃
            continue
        visited.add(v)
        mst.append((u, v, w))
        total += w
        if verbose:
            print(f"    加入边 {u}-{v}（权重 {w}），当前树权重和 = {total}")
        for nxt, nw in graph[v]:
            if nxt not in visited:
                heapq.heappush(pq, (nw, v, nxt))
    return mst, total


# ======================================================================
# 5. Kruskal —— 最小生成树，把所有边按权重排序后贪心地加不成环的边
#    复杂度：O(E log E)，瓶颈在排序。适合稀疏图。
#    需要并查集（Union-Find）判断两点是否已连通。
# ======================================================================
class UnionFind:
    """并查集：路径压缩 + 按秩合并，单次操作近似 O(1)。"""

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
            return False               # 已连通，加这条边会成环
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def kruskal(vertices, edges, verbose=True):
    uf = UnionFind(vertices)
    mst, total = [], 0
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        if uf.union(u, v):
            mst.append((u, v, w))
            total += w
            if verbose:
                print(f"    接受边 {u}-{v}（权重 {w}），累计 = {total}")
        elif verbose:
            print(f"    丢弃边 {u}-{v}（权重 {w}）：会形成环")
    return mst, total


def undirected_edge_list(graph):
    """把邻接表转成去重后的无向边列表 [(u, v, w), ...]。"""
    seen = set()
    edges = []
    for u, nbrs in graph.items():
        for v, w in nbrs:
            key = tuple(sorted((u, v)))
            if key not in seen:
                seen.add(key)
                edges.append((u, v, w))
    return edges


if __name__ == "__main__":
    print("=" * 68)
    print("【1】Dijkstra 单源最短路（逐步松弛过程）")
    print("=" * 68)
    dist, prev = dijkstra(GRAPH, "A")
    print()
    for target in ["D", "E", "F"]:
        path = build_path(prev, "A", target)
        print(f"  A -> {target}：距离 {dist[target]}，路径 {' -> '.join(path)}")

    print()
    print("=" * 68)
    print("【2】Bellman-Ford（含负权边）")
    print("=" * 68)
    # 有向图，含负权边但无负环
    vs = ["S", "A", "B", "C", "D"]
    es = [("S", "A", 4), ("S", "B", 5), ("A", "C", 3),
          ("B", "A", -3), ("B", "D", 2), ("C", "D", -1)]
    print("  有向边：", es)
    d, has_neg = bellman_ford(vs, es, "S")
    print(f"  存在负环？{has_neg}")
    print("  最短距离：", {v: fmt(d[v]) for v in vs})

    print()
    print("  再构造一个含负环的图：")
    es_neg = es + [("D", "B", -4)]      # B->D(2) + D->B(-4) = -2，形成负环
    d2, has_neg2 = bellman_ford(vs, es_neg, "S", verbose=False)
    print(f"  存在负环？{has_neg2} （B -> D -> B 权重和为 -2）")

    print()
    print("=" * 68)
    print("【3】Floyd-Warshall 全源最短路")
    print("=" * 68)
    verts = sorted(GRAPH)
    # 无向图拆成两条有向边
    directed = []
    for u, v, w in undirected_edge_list(GRAPH):
        directed.append((u, v, w))
        directed.append((v, u, w))
    mat, idx = floyd_warshall(verts, directed)
    print("      " + "".join(f"{v:>6}" for v in verts))
    for u in verts:
        row = "".join(f"{fmt(mat[idx[u]][idx[v]]):>6}" for v in verts)
        print(f"  {u:<4}" + row)

    print()
    print("=" * 68)
    print("【4】Prim 最小生成树（从 A 出发）")
    print("=" * 68)
    mst_p, tot_p = prim(GRAPH, "A")
    print(f"  MST 边集 = {mst_p}")
    print(f"  总权重 = {tot_p}")

    print()
    print("=" * 68)
    print("【5】Kruskal 最小生成树（按边权从小到大）")
    print("=" * 68)
    mst_k, tot_k = kruskal(sorted(GRAPH), undirected_edge_list(GRAPH))
    print(f"  MST 边集 = {mst_k}")
    print(f"  总权重 = {tot_k}")
    print()
    print(f"  Prim 与 Kruskal 总权重是否相等？{tot_p == tot_k}")
    print("  （MST 可能不唯一，但最小总权重一定唯一）")
