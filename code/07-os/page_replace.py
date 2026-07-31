# -*- coding: utf-8 -*-
"""页面置换算法对比：FIFO / LRU / OPT / Clock，并复现 Belady 异常。

缺页率 = 缺页次数 / 访问总次数。只依赖标准库。
"""

from collections import OrderedDict, deque


def fifo(pages, frames):
    """先进先出：淘汰驻留内存最久的页。"""
    mem, queue, faults = set(), deque(), 0
    for p in pages:
        if p not in mem:
            faults += 1
            if len(mem) == frames:
                mem.remove(queue.popleft())   # 淘汰最早进入的页
            mem.add(p)
            queue.append(p)
    return faults


def lru(pages, frames):
    """最近最久未使用：淘汰上次访问距今最远的页。用 OrderedDict 模拟栈。"""
    mem, faults = OrderedDict(), 0
    for p in pages:
        if p in mem:
            mem.move_to_end(p)                # 命中：移到最近端
        else:
            faults += 1
            if len(mem) == frames:
                mem.popitem(last=False)       # 淘汰最久未使用（最旧端）
            mem[p] = True
    return faults


def opt(pages, frames):
    """最佳置换（OPT）：淘汰将来最晚被访问（或不再访问）的页。理论下界。"""
    mem, faults = set(), 0
    for i, p in enumerate(pages):
        if p in mem:
            continue
        faults += 1
        if len(mem) == frames:
            # 计算每个驻留页下次被访问的位置，取最远者淘汰
            def next_use(page):
                for j in range(i + 1, len(pages)):
                    if pages[j] == page:
                        return j
                return float("inf")           # 不再访问 → 最优淘汰对象
            mem.remove(max(mem, key=next_use))
        mem.add(p)
    return faults


def clock(pages, frames):
    """时钟算法（二次机会）：环形扫描，引用位为 1 则清零放过，为 0 则淘汰。"""
    slots = [None] * frames                   # 页框环
    refbit = [0] * frames                     # 引用位
    hand, faults = 0, 0
    pos = {}                                  # 页 -> 页框下标
    for p in pages:
        if p in pos:
            refbit[pos[p]] = 1                # 命中：置引用位
            continue
        faults += 1
        while refbit[hand] == 1:              # 引用位 1：给二次机会
            refbit[hand] = 0
            hand = (hand + 1) % frames
        if slots[hand] is not None:
            del pos[slots[hand]]              # 淘汰指针所指页
        slots[hand] = p
        pos[p] = hand
        refbit[hand] = 1
        hand = (hand + 1) % frames
    return faults


ALGOS = [("FIFO", fifo), ("LRU", lru), ("Clock", clock), ("OPT", opt)]


def compare(pages, frame_list):
    print(f"访问串: {' '.join(map(str, pages))}  (共 {len(pages)} 次访问)")
    header = "页框数 " + "".join(f"{name:>8}" for name, _ in ALGOS)
    print(header)
    for f in frame_list:
        row = f"{f:>5}  "
        for _, algo in ALGOS:
            faults = algo(pages, f)
            row += f"{faults:>6}缺"
        print(row)
    print()


if __name__ == "__main__":
    print("===== 常规对比：OPT 最少，LRU 接近 OPT，FIFO 最差 =====")
    seq = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
    compare(seq, [3, 4])

    print("===== Belady 异常复现：FIFO 页框 3->4 缺页反而增多 =====")
    belady = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    compare(belady, [3, 4])
    f3, f4 = fifo(belady, 3), fifo(belady, 4)
    print(f"FIFO: 3 页框缺页 {f3} 次，4 页框缺页 {f4} 次 -> "
          f"{'出现 Belady 异常!' if f4 > f3 else '未出现异常'}")
    l3, l4 = lru(belady, 3), lru(belady, 4)
    print(f"LRU : 3 页框缺页 {l3} 次，4 页框缺页 {l4} 次 -> "
          "LRU 是栈式算法，页框增加缺页数单调不增，不会出现 Belady 异常")
