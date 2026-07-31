"""
回溯与分支限界
==============
用 N 皇后、子集和、全排列三个问题演示回溯法的通用框架，
并通过"剪枝 vs 不剪枝"的节点数对比，让学生直观感受剪枝的威力。

回溯法通用框架：
    def backtrack(路径, 选择列表):
        if 满足结束条件:
            记录结果; return
        for 选择 in 选择列表:
            if 不满足约束: continue      # <- 剪枝就发生在这里
            做出选择
            backtrack(新路径, 新选择列表)
            撤销选择                      # <- 回溯

运行方式：
    python nqueens.py
"""

import time


# ======================================================================
# 1. N 皇后：暴力枚举 vs 回溯剪枝
# ======================================================================
def solve_n_queens_bruteforce(n):
    """把每行的皇后列号看成一个 n 位 n 进制数，全部枚举后再检查。
    搜索空间 n^n，n=8 时是 16777216 种，非常浪费。
    """
    count = [0]      # 解的个数
    nodes = [0]      # 生成的完整方案数（衡量工作量）

    def check(cols):
        for i in range(n):
            for j in range(i + 1, n):
                if cols[i] == cols[j] or abs(cols[i] - cols[j]) == j - i:
                    return False
        return True

    def enumerate_all(cols):
        if len(cols) == n:
            nodes[0] += 1
            if check(cols):
                count[0] += 1
            return
        for c in range(n):
            enumerate_all(cols + [c])

    enumerate_all([])
    return count[0], nodes[0]


def solve_n_queens(n, collect=True):
    """回溯 + 剪枝：用三个集合 O(1) 判断冲突。
    - cols：已占用的列
    - diag1：主对角线，同一条上 (row - col) 相同
    - diag2：副对角线，同一条上 (row + col) 相同
    返回 (解的列表, 访问的搜索树结点数)
    """
    solutions = []
    cols, diag1, diag2 = set(), set(), set()
    placement = []
    nodes = [0]

    def backtrack(row):
        nodes[0] += 1                       # 统计进入了多少个搜索树结点
        if row == n:
            if collect:
                solutions.append(placement[:])
            else:
                solutions.append(None)
            return
        for c in range(n):
            # 剪枝：任何一个冲突条件成立，这整棵子树都不用再搜了
            if c in cols or (row - c) in diag1 or (row + c) in diag2:
                continue
            cols.add(c)
            diag1.add(row - c)
            diag2.add(row + c)
            placement.append(c)

            backtrack(row + 1)

            placement.pop()                 # 撤销选择（回溯）
            cols.remove(c)
            diag1.remove(row - c)
            diag2.remove(row + c)

    backtrack(0)
    return solutions, nodes[0]


def print_board(cols):
    """把一维列号数组画成棋盘，Q 表示皇后。"""
    n = len(cols)
    for r in range(n):
        print("    " + " ".join("Q" if cols[r] == c else "." for c in range(n)))


# ======================================================================
# 2. 子集和问题（Subset Sum）：判断能否选出若干数使其和为 target
#    这是一个 NP 完全问题，回溯 + 剪枝是常用的实用解法。
# ======================================================================
def subset_sum(nums, target, use_pruning=True):
    """返回 (所有可行子集, 访问结点数)。
    剪枝策略（先把 nums 降序排序效果更好）：
      - 上界剪枝：当前和 + 剩余全部 < target，不可能达成，直接返回
      - 下界剪枝：当前和 > target（全为正数时），也不可能，直接返回
    """
    nums = sorted(nums, reverse=True)
    suffix = [0] * (len(nums) + 1)          # suffix[i] = nums[i:] 的总和
    for i in range(len(nums) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + nums[i]

    results = []
    path = []
    nodes = [0]

    def backtrack(i, cur):
        nodes[0] += 1
        if cur == target:
            results.append(path[:])
            return
        if i == len(nums):
            return
        if use_pruning:
            if cur > target:                    # 下界剪枝
                return
            if cur + suffix[i] < target:        # 上界剪枝
                return

        path.append(nums[i])                    # 选 nums[i]
        backtrack(i + 1, cur + nums[i])
        path.pop()

        backtrack(i + 1, cur)                   # 不选 nums[i]

    backtrack(0, 0)
    return results, nodes[0]


# ======================================================================
# 3. 全排列：回溯框架最基础的应用
# ======================================================================
def permutations(items):
    """生成 items 的全排列，共 n! 个。"""
    result = []
    used = [False] * len(items)
    path = []

    def backtrack():
        if len(path) == len(items):
            result.append(path[:])
            return
        for i, x in enumerate(items):
            if used[i]:
                continue
            used[i] = True
            path.append(x)
            backtrack()
            path.pop()
            used[i] = False

    backtrack()
    return result


# ======================================================================
# 4. 分支限界思想演示：用"当前最优解"作为界，进一步砍掉无望分支
#    问题：从若干物品中选出总重不超过 cap 的组合，使价值最大（0-1 背包）
# ======================================================================
def knapsack_branch_and_bound(weights, values, cap):
    """按"单位价值"降序排序，用贪心松弛解作为上界（bound）。
    如果某分支的上界还不如已找到的最优解，直接剪掉。
    """
    n = len(weights)
    order = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)
    w = [weights[i] for i in order]
    v = [values[i] for i in order]

    best = [0]
    nodes = [0]

    def bound(i, cur_w, cur_v):
        """分数背包（允许切割）的最优解，一定 >= 0-1 背包的最优解，是合法上界。"""
        total, rem = cur_v, cap - cur_w
        j = i
        while j < n and w[j] <= rem:
            total += v[j]
            rem -= w[j]
            j += 1
        if j < n:
            total += v[j] * rem / w[j]      # 装入一部分
        return total

    def dfs(i, cur_w, cur_v):
        nodes[0] += 1
        if cur_v > best[0]:
            best[0] = cur_v
        if i == n:
            return
        if bound(i, cur_w, cur_v) <= best[0]:   # 限界剪枝
            return
        if cur_w + w[i] <= cap:
            dfs(i + 1, cur_w + w[i], cur_v + v[i])
        dfs(i + 1, cur_w, cur_v)

    dfs(0, 0, 0)
    return best[0], nodes[0]


if __name__ == "__main__":
    print("=" * 68)
    print("【1】N 皇后：一个 8 皇后解")
    print("=" * 68)
    sols, nodes = solve_n_queens(8)
    print(f"  8 皇后共有 {len(sols)} 组解，搜索树访问结点数 = {nodes}")
    print("  第一组解（列号数组 =", sols[0], "）：")
    print_board(sols[0])

    print()
    print("=" * 68)
    print("【2】剪枝的威力：暴力枚举 vs 回溯剪枝")
    print("=" * 68)
    print(f"  {'n':<4}{'解的个数':<10}{'暴力枚举工作量':<18}{'回溯访问结点':<15}{'节省倍数'}")
    print("  " + "-" * 62)
    for n in range(4, 9):
        t0 = time.perf_counter()
        cnt_bf, nodes_bf = (None, None)
        if n <= 7:                          # n=8 暴力枚举太慢，跳过
            cnt_bf, nodes_bf = solve_n_queens_bruteforce(n)
        t_bf = time.perf_counter() - t0

        t0 = time.perf_counter()
        sols_n, nodes_n = solve_n_queens(n, collect=False)
        t_bt = time.perf_counter() - t0

        if nodes_bf is not None:
            assert cnt_bf == len(sols_n), "两种方法解的个数应该一致"
            ratio = f"{nodes_bf / nodes_n:.1f}x"
            bf_str = f"{nodes_bf}"
        else:
            ratio = "-"
            bf_str = f"{8 ** 8}(未跑)"
        print(f"  {n:<4}{len(sols_n):<10}{bf_str:<18}{nodes_n:<15}{ratio}")
    print()
    print("  说明：暴力枚举工作量 = n^n 个完整方案；回溯只访问约 n! 量级以下的结点，")
    print("        剪枝越早，砍掉的子树越大。")

    print()
    print("=" * 68)
    print("【3】子集和问题：剪枝 vs 不剪枝")
    print("=" * 68)
    nums = [3, 34, 4, 12, 5, 2, 7, 8, 11, 19]
    target = 30
    res_p, nodes_p = subset_sum(nums, target, use_pruning=True)
    res_np, nodes_np = subset_sum(nums, target, use_pruning=False)
    print(f"  集合 = {nums}，目标 = {target}")
    print(f"  可行子集共 {len(res_p)} 个，例如：{res_p[:3]}")
    print(f"  开启剪枝：访问结点 {nodes_p}")
    print(f"  关闭剪枝：访问结点 {nodes_np}")
    print(f"  剪枝减少了 {(1 - nodes_p / nodes_np) * 100:.1f}% 的搜索量")
    assert len(res_p) == len(res_np), "剪枝不能改变解集"

    print()
    print("=" * 68)
    print("【4】全排列")
    print("=" * 68)
    perms = permutations(["A", "B", "C"])
    print(f"  ['A','B','C'] 的全排列共 {len(perms)} 个：")
    print("   ", ["".join(p) for p in perms])

    print()
    print("=" * 68)
    print("【5】分支限界解 0-1 背包")
    print("=" * 68)
    wts = [7, 3, 4, 5, 2, 9, 6, 8]
    vals = [42, 12, 40, 25, 8, 50, 30, 44]
    cap = 20
    best, nodes_bb = knapsack_branch_and_bound(wts, vals, cap)
    print(f"  重量 = {wts}")
    print(f"  价值 = {vals}，容量 = {cap}")
    print(f"  最优价值 = {best}，分支限界访问结点数 = {nodes_bb}")
    print(f"  （不剪枝的完全枚举需要 2^{len(wts)} = {2 ** len(wts)} 个叶子结点）")
