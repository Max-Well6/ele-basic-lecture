# -*- coding: utf-8 -*-
"""银行家算法：安全性检查 + 资源请求判定。

Need = Max - Allocation
安全性算法：反复寻找 Need <= Work 的进程，模拟其执行完毕并回收资源，
若所有进程都能完成则系统处于安全状态。只依赖标准库。
"""


def safety_check(available, alloc, need, names):
    """返回 (是否安全, 安全序列)。"""
    n = len(names)
    work = available[:]                # 当前可分配资源向量
    finish = [False] * n
    sequence = []
    while len(sequence) < n:
        found = False
        for i in range(n):
            if finish[i]:
                continue
            if all(need[i][j] <= work[j] for j in range(len(work))):
                # 进程 i 可以拿到所需全部资源 → 执行完毕后归还 Allocation
                work = [work[j] + alloc[i][j] for j in range(len(work))]
                finish[i] = True
                sequence.append(names[i])
                print(f"    {names[i]} 可完成，归还后 Work = {work}")
                found = True
        if not found:                  # 一轮下来没人能满足 → 不安全
            return False, sequence
    return True, sequence


def request(pid, req, available, alloc, need, names):
    """进程 pid 请求资源向量 req，判定能否批准。"""
    i = names.index(pid)
    print(f"\n>> {pid} 请求 {req}")
    if any(req[j] > need[i][j] for j in range(len(req))):
        print("   拒绝：请求超过声明的最大需求 Need")
        return False
    if any(req[j] > available[j] for j in range(len(req))):
        print("   暂缓：资源不足，进程需等待")
        return False
    # 试探性分配
    avail2 = [available[j] - req[j] for j in range(len(req))]
    alloc2 = [row[:] for row in alloc]
    need2 = [row[:] for row in need]
    for j in range(len(req)):
        alloc2[i][j] += req[j]
        need2[i][j] -= req[j]
    print("   试探分配后做安全性检查：")
    safe, seq = safety_check(avail2, alloc2, need2, names)
    if safe:
        print(f"   批准！安全序列: {' -> '.join(seq)}")
        available[:] = avail2
        alloc[:] = alloc2
        need[:] = need2
        return True
    print("   拒绝：分配后系统将进入不安全状态，回滚")
    return False


if __name__ == "__main__":
    # 经典例子：5 个进程，3 类资源 (A, B, C)，总量 (10, 5, 7)
    names = ["P0", "P1", "P2", "P3", "P4"]
    max_ = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
    alloc = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
    available = [3, 3, 2]
    need = [[max_[i][j] - alloc[i][j] for j in range(3)] for i in range(5)]

    print("进程   Max      Alloc    Need")
    for i, name in enumerate(names):
        print(f"{name}   {max_[i]}  {alloc[i]}  {need[i]}")
    print(f"Available = {available}")

    print("\n===== 初始状态安全性检查 =====")
    safe, seq = safety_check(available, alloc, need, names)
    print(f"结论：{'安全' if safe else '不安全'}，安全序列: {' -> '.join(seq)}")

    # 请求 1：P1 请求 (1,0,2) —— 教科书上应批准
    request("P1", [1, 0, 2], available, alloc, need, names)

    # 请求 2：P4 请求 (3,3,0) —— 超过 Available，应暂缓
    request("P4", [3, 3, 0], available, alloc, need, names)

    # 请求 3：P0 请求 (0,2,0) —— 分配后不安全，应拒绝并回滚
    request("P0", [0, 2, 0], available, alloc, need, names)

    print(f"\n最终 Available = {available}（P0 的请求已回滚，状态保持安全）")
