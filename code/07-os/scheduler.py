# -*- coding: utf-8 -*-
"""CPU 调度算法模拟：FCFS / SJF / RR / 优先级 / 多级反馈队列（MLFQ）。

对同一组进程分别运行五种调度算法，输出文本甘特图与
周转时间 / 等待时间对比表。只依赖标准库。

周转时间 = 完成时间 - 到达时间
等待时间 = 周转时间 - 运行时间
"""

from collections import deque


class Proc:
    def __init__(self, name, arrive, burst, priority=0):
        self.name = name          # 进程名
        self.arrive = arrive      # 到达时间
        self.burst = burst        # 总运行时间
        self.priority = priority  # 优先级（数字越小优先级越高）
        self.remain = burst       # 剩余运行时间
        self.finish = None        # 完成时间

    def reset(self):
        self.remain = self.burst
        self.finish = None


# ---------------------------------------------------------------- 调度算法
def fcfs(procs):
    """先来先服务：按到达时间排队，非抢占。返回甘特图段列表 [(name, start, end)]。"""
    gantt, t = [], 0
    for p in sorted(procs, key=lambda p: p.arrive):
        t = max(t, p.arrive)               # CPU 可能空闲等待下一个进程到达
        gantt.append((p.name, t, t + p.burst))
        t += p.burst
        p.finish = t
    return gantt


def sjf(procs):
    """短作业优先（非抢占）：就绪队列中选剩余时间最短者。"""
    gantt, t, done = [], 0, 0
    pending = sorted(procs, key=lambda p: p.arrive)
    ready = []
    while done < len(procs):
        while pending and pending[0].arrive <= t:
            ready.append(pending.pop(0))
        if not ready:                       # 就绪队列为空，快进到下一个到达时刻
            t = pending[0].arrive
            continue
        ready.sort(key=lambda p: p.burst)   # 选最短作业
        p = ready.pop(0)
        gantt.append((p.name, t, t + p.burst))
        t += p.burst
        p.finish = t
        done += 1
    return gantt


def rr(procs, q=2):
    """时间片轮转：时间片 q，队尾追加新到达进程。"""
    gantt, t = [], 0
    pending = sorted(procs, key=lambda p: p.arrive)
    queue = deque()
    while pending or queue:
        while pending and pending[0].arrive <= t:
            queue.append(pending.pop(0))
        if not queue:
            t = pending[0].arrive
            continue
        p = queue.popleft()
        run = min(q, p.remain)
        gantt.append((p.name, t, t + run))
        t += run
        p.remain -= run
        while pending and pending[0].arrive <= t:   # 运行期间新到达者先入队
            queue.append(pending.pop(0))
        if p.remain > 0:
            queue.append(p)                          # 未跑完，回到队尾
        else:
            p.finish = t
    return gantt


def priority_sched(procs):
    """优先级调度（非抢占）：就绪队列中选优先级数字最小者。"""
    gantt, t, done = [], 0, 0
    pending = sorted(procs, key=lambda p: p.arrive)
    ready = []
    while done < len(procs):
        while pending and pending[0].arrive <= t:
            ready.append(pending.pop(0))
        if not ready:
            t = pending[0].arrive
            continue
        ready.sort(key=lambda p: p.priority)
        p = ready.pop(0)
        gantt.append((p.name, t, t + p.burst))
        t += p.burst
        p.finish = t
        done += 1
    return gantt


def mlfq(procs, quanta=(1, 2, 4)):
    """多级反馈队列：3 级队列，时间片 1/2/4，用完时间片降级，最低级轮转。"""
    gantt, t = [], 0
    pending = sorted(procs, key=lambda p: p.arrive)
    queues = [deque() for _ in quanta]

    def enqueue_arrivals(now):
        while pending and pending[0].arrive <= now:
            queues[0].append(pending.pop(0))         # 新进程进最高级队列

    enqueue_arrivals(t)
    while pending or any(queues):
        lvl = next((i for i, qu in enumerate(queues) if qu), None)
        if lvl is None:                              # 全部队列空，快进
            t = pending[0].arrive
            enqueue_arrivals(t)
            continue
        p = queues[lvl].popleft()
        run = min(quanta[lvl], p.remain)
        gantt.append((p.name, t, t + run))
        t += run
        p.remain -= run
        enqueue_arrivals(t)
        if p.remain == 0:
            p.finish = t
        else:                                        # 用完时间片仍未结束 → 降级
            queues[min(lvl + 1, len(queues) - 1)].append(p)
    return gantt


# ---------------------------------------------------------------- 输出工具
def merge_gantt(gantt):
    """合并相邻同名段，便于阅读。"""
    merged = []
    for seg in gantt:
        if merged and merged[-1][0] == seg[0] and merged[-1][2] == seg[1]:
            merged[-1] = (seg[0], merged[-1][1], seg[2])
        else:
            merged.append(list(seg))
    return [tuple(s) for s in merged]


def print_gantt(gantt):
    gantt = merge_gantt(gantt)
    line = " | ".join(f"{name}:{s}-{e}" for name, s, e in gantt)
    print("  甘特图: |", line, "|")


def report(title, algo, procs, **kw):
    for p in procs:
        p.reset()
    gantt = algo(procs, **kw) if kw else algo(procs)
    print(f"\n[{title}]")
    print_gantt(gantt)
    print(f"  {'进程':<4} {'到达':>4} {'运行':>4} {'完成':>4} {'周转':>4} {'等待':>4}")
    tot_tat = tot_wait = 0
    for p in sorted(procs, key=lambda p: p.name):
        tat = p.finish - p.arrive          # 周转时间
        wait = tat - p.burst               # 等待时间
        tot_tat += tat
        tot_wait += wait
        print(f"  {p.name:<5} {p.arrive:>4} {p.burst:>4} {p.finish:>4} {tat:>4} {wait:>4}")
    n = len(procs)
    print(f"  平均周转时间 = {tot_tat / n:.2f}，平均等待时间 = {tot_wait / n:.2f}")
    return tot_tat / n, tot_wait / n


if __name__ == "__main__":
    procs = [
        Proc("P1", 0, 7, priority=3),
        Proc("P2", 2, 4, priority=1),
        Proc("P3", 4, 1, priority=4),
        Proc("P4", 5, 4, priority=2),
    ]
    print("进程集：P1(0到达,7) P2(2,4) P3(4,1) P4(5,4)，优先级 P2>P4>P1>P3")

    summary = {}
    summary["FCFS"] = report("FCFS 先来先服务", fcfs, procs)
    summary["SJF"] = report("SJF 短作业优先(非抢占)", sjf, procs)
    summary["RR(q=2)"] = report("RR 时间片轮转 q=2", rr, procs, q=2)
    summary["优先级"] = report("优先级调度(非抢占)", priority_sched, procs)
    summary["MLFQ"] = report("多级反馈队列 时间片1/2/4", mlfq, procs)

    print("\n===== 五种算法对比 =====")
    print(f"{'算法':<10} {'平均周转':>8} {'平均等待':>8}")
    for name, (tat, wait) in summary.items():
        print(f"{name:<10} {tat:>8.2f} {wait:>8.2f}")
    print("结论：SJF 平均等待最短（理论最优）；RR 响应快但周转变长；MLFQ 折中。")
