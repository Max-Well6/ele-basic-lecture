"""
动态规划合集
============
包含 5 个经典 DP 问题的教学实现，重点展示"状态定义 -> 转移方程 -> 边界"三段式思路，
并把 DP 表格逐行打印出来，让抽象的递推过程可视化。

运行方式：
    python dp.py
"""

from functools import lru_cache


# ======================================================================
# 1. 斐波那契：从指数级递归到 O(n) DP，理解"重叠子问题"
# ======================================================================
def fib_naive(n, counter):
    """朴素递归：T(n) = T(n-1) + T(n-2) + O(1)，约等于 O(1.618^n)，指数爆炸。"""
    counter[0] += 1
    if n < 2:
        return n
    return fib_naive(n - 1, counter) + fib_naive(n - 2, counter)


def fib_dp(n):
    """自底向上递推，滚动变量把空间从 O(n) 压到 O(1)。
    状态定义：f[i] = 第 i 个斐波那契数
    转移方程：f[i] = f[i-1] + f[i-2]
    边界条件：f[0] = 0, f[1] = 1
    """
    if n < 2:
        return n
    prev, cur = 0, 1
    for _ in range(2, n + 1):
        prev, cur = cur, prev + cur
    return cur


@lru_cache(maxsize=None)
def fib_memo(n):
    """记忆化搜索：自顶向下 + 缓存，写法最接近原始递归定义。"""
    if n < 2:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)


# ======================================================================
# 2. 0-1 背包
#    状态定义：dp[i][w] = 只考虑前 i 件物品、容量上限为 w 时的最大价值
#    转移方程：dp[i][w] = max(dp[i-1][w],                     不选第 i 件
#                            dp[i-1][w - wt[i]] + val[i])    选第 i 件（需 w >= wt[i]）
#    边界条件：dp[0][w] = 0（没有物品，价值为 0）
# ======================================================================
def knapsack_01(weights, values, capacity, verbose=True):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]                          # 先假设不选
            if w >= weights[i - 1]:
                cand = dp[i - 1][w - weights[i - 1]] + values[i - 1]
                if cand > dp[i][w]:
                    dp[i][w] = cand                          # 选更优的方案

    if verbose:
        print("  DP 表（行=前 i 件物品，列=容量 w）：")
        print("       " + "".join(f"{w:4d}" for w in range(capacity + 1)))
        for i in range(n + 1):
            tag = "  空 " if i == 0 else f" 物{i} "
            print(f"  {tag}" + "".join(f"{v:4d}" for v in dp[i]))

    # 回溯求解具体选了哪些物品：从 dp[n][capacity] 反推
    chosen, w = [], capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:      # 值发生变化，说明第 i 件被选中
            chosen.append(i - 1)
            w -= weights[i - 1]
    chosen.reverse()
    return dp[n][capacity], chosen


def knapsack_01_rolling(weights, values, capacity):
    """一维滚动数组版：容量必须倒序遍历，否则第 i 件物品会被重复使用。"""
    dp = [0] * (capacity + 1)
    for wt, val in zip(weights, values):
        for w in range(capacity, wt - 1, -1):    # 倒序！
            dp[w] = max(dp[w], dp[w - wt] + val)
    return dp[capacity]


# ======================================================================
# 3. 最长公共子序列 LCS
#    状态定义：dp[i][j] = s1 前 i 个字符与 s2 前 j 个字符的 LCS 长度
#    转移方程：s1[i-1] == s2[j-1] -> dp[i][j] = dp[i-1][j-1] + 1
#              否则               -> dp[i][j] = max(dp[i-1][j], dp[i][j-1])
#    边界条件：dp[0][*] = dp[*][0] = 0
# ======================================================================
def lcs(s1, s2, verbose=True):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    if verbose:
        print(f"  s1 = {s1!r}, s2 = {s2!r}")
        print("        ''" + "".join(f"{c:>4}" for c in s2))
        for i in range(m + 1):
            head = "  ''" if i == 0 else f"  {s1[i - 1]} "
            print(head + "".join(f"{v:4d}" for v in dp[i]))

    # 回溯还原一条 LCS
    i, j, out = m, n, []
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            out.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return dp[m][n], "".join(reversed(out))


# ======================================================================
# 4. 编辑距离（Levenshtein Distance）—— 招牌案例，逐行打印 DP 表
#    状态定义：dp[i][j] = 把 s1 前 i 个字符变成 s2 前 j 个字符的最少操作数
#    转移方程：s1[i-1] == s2[j-1] -> dp[i][j] = dp[i-1][j-1]
#              否则 dp[i][j] = 1 + min(dp[i-1][j],    删除 s1[i-1]
#                                      dp[i][j-1],    插入 s2[j-1]
#                                      dp[i-1][j-1])  替换
#    边界条件：dp[i][0] = i（全删），dp[0][j] = j（全插）
# ======================================================================
def edit_distance(s1, s2, verbose=True):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    if verbose:
        print(f"  求 {s1!r} -> {s2!r} 的编辑距离")
        print("  逐行填表过程（行标题为 s1 的字符，列标题为 s2 的字符）：")
        print("         ''" + "".join(f"{c:>4}" for c in s2))
        print("     ''" + "".join(f"{v:4d}" for v in dp[0]))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]                 # 字符相同，无需操作
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],            # 删除
                                   dp[i][j - 1],            # 插入
                                   dp[i - 1][j - 1])        # 替换
        if verbose:
            print(f"     {s1[i - 1]} " + "".join(f"{v:4d}" for v in dp[i]))

    # 回溯还原操作序列，让学生看清"最少 k 步"到底是哪 k 步
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(f"替换 位置{i} 的 '{s1[i - 1]}' 为 '{s2[j - 1]}'")
            i, j = i - 1, j - 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(f"在位置{i}后 插入 '{s2[j - 1]}'")
            j -= 1
        else:
            ops.append(f"删除 位置{i} 的 '{s1[i - 1]}'")
            i -= 1
    ops.reverse()
    return dp[m][n], ops


# ======================================================================
# 5. 最长递增子序列 LIS：O(n^2) DP 与 O(n log n) 贪心+二分
#    状态定义：dp[i] = 以 a[i] 结尾的最长递增子序列长度
#    转移方程：dp[i] = max(dp[j] + 1)，其中 j < i 且 a[j] < a[i]
#    边界条件：dp[i] = 1（每个元素自身构成长度 1 的序列）
# ======================================================================
def lis_dp(a):
    if not a:
        return 0, []
    n = len(a)
    dp = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if a[j] < a[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j
    best = max(range(n), key=lambda i: dp[i])
    seq, k = [], best
    while k != -1:
        seq.append(a[k])
        k = prev[k]
    seq.reverse()
    return dp[best], seq


def lis_binary(a):
    """贪心 + 二分：tails[k] 记录长度为 k+1 的递增子序列的最小可能结尾。O(n log n)"""
    import bisect
    tails = []
    for x in a:
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x          # 用更小的结尾替换，为后续留更多空间
    return len(tails)


if __name__ == "__main__":
    print("=" * 66)
    print("【1】斐波那契：递归 vs 记忆化 vs 递推")
    print("=" * 66)
    counter = [0]
    print("  fib_naive(30) =", fib_naive(30, counter), f"（递归调用 {counter[0]} 次）")
    print("  fib_memo(30)  =", fib_memo(30), "（记忆化，调用次数 O(n)）")
    print("  fib_dp(30)    =", fib_dp(30), "（递推，O(n) 时间 O(1) 空间）")
    print("  fib_dp(200)   =", fib_dp(200), "（Python 大整数，直接算）")

    print()
    print("=" * 66)
    print("【2】0-1 背包")
    print("=" * 66)
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    cap = 8
    print(f"  物品重量 {weights}，价值 {values}，背包容量 {cap}")
    best, chosen = knapsack_01(weights, values, cap)
    print(f"  最大价值 = {best}")
    print(f"  选择物品下标 = {chosen}，"
          f"总重 = {sum(weights[i] for i in chosen)}，"
          f"总价值 = {sum(values[i] for i in chosen)}")
    print(f"  一维滚动数组结果 = {knapsack_01_rolling(weights, values, cap)}（应与上面一致）")

    print()
    print("=" * 66)
    print("【3】最长公共子序列 LCS")
    print("=" * 66)
    length, seq = lcs("ABCBDAB", "BDCABA")
    print(f"  LCS 长度 = {length}，一条 LCS = {seq!r}")

    print()
    print("=" * 66)
    print("【4】编辑距离（招牌案例：DP 表逐行打印）")
    print("=" * 66)
    dist, ops = edit_distance("kitten", "sitting")
    print(f"  编辑距离 = {dist}")
    print("  一种最优操作序列：")
    for k, op in enumerate(ops, 1):
        print(f"    {k}. {op}")

    print()
    print("  再看一个中文例子：")
    dist2, ops2 = edit_distance("算法分析", "算法设计")
    print(f"  编辑距离 = {dist2}，操作：{ops2}")

    print()
    print("=" * 66)
    print("【5】最长递增子序列 LIS")
    print("=" * 66)
    arr = [10, 9, 2, 5, 3, 7, 101, 18, 4, 8, 12]
    n1, s1 = lis_dp(arr)
    print(f"  数组 = {arr}")
    print(f"  O(n^2) DP：长度 = {n1}，序列 = {s1}")
    print(f"  O(n log n) 贪心+二分：长度 = {lis_binary(arr)}（应与上面长度一致）")

    print()
    print("小结：DP 的核心是三件事 —— 状态定义、转移方程、边界条件。")
    print("      写不出转移方程，往往是状态定义得不够'完备'或不够'无后效性'。")
