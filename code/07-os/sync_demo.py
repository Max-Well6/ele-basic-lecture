# -*- coding: utf-8 -*-
"""进程同步三大经典问题的 threading 演示：
1) 生产者-消费者（信号量 + 互斥锁）
2) 读者-写者（读者优先，读计数器）
3) 哲学家就餐（资源分级避免死锁）

所有线程均有限循环，程序保证自动结束。只依赖标准库。
"""

import threading
import time
import random

random.seed(42)   # 固定随机种子，输出可复现


# ================================================= 1. 生产者-消费者
def producer_consumer():
    print("===== 1. 生产者-消费者（缓冲区容量 3，各生产/消费 6 件）=====")
    BUF_SIZE, N_ITEM = 3, 6
    buf = []
    mutex = threading.Lock()                  # 保护缓冲区
    empty = threading.Semaphore(BUF_SIZE)     # 空槽位数量
    full = threading.Semaphore(0)             # 产品数量

    def producer():
        for i in range(N_ITEM):
            item = f"item{i}"
            empty.acquire()                   # P(empty)：等空位
            with mutex:
                buf.append(item)
                print(f"  生产 {item}，缓冲区 {buf}")
            full.release()                    # V(full)：产品 +1
            time.sleep(random.uniform(0, 0.02))

    def consumer():
        for _ in range(N_ITEM):
            full.acquire()                    # P(full)：等产品
            with mutex:
                item = buf.pop(0)
                print(f"  消费 {item}，缓冲区 {buf}")
            empty.release()                   # V(empty)：空位 +1
            time.sleep(random.uniform(0, 0.03))

    ts = [threading.Thread(target=producer), threading.Thread(target=consumer)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)
    print("  生产者-消费者结束，缓冲区剩余:", buf)


# ================================================= 2. 读者-写者
def reader_writer():
    print("\n===== 2. 读者-写者（读者优先，3 读者 2 写者）=====")
    rw_lock = threading.Lock()        # 写锁：写者独占，第一个读者抢占
    count_lock = threading.Lock()     # 保护 read_count
    state = {"read_count": 0, "value": 0}

    def reader(rid):
        for _ in range(2):
            with count_lock:
                state["read_count"] += 1
                if state["read_count"] == 1:
                    rw_lock.acquire()             # 第一个读者锁住写者
            print(f"  读者{rid} 读到 value={state['value']}"
                  f"（当前 {state['read_count']} 个读者）")
            time.sleep(0.01)
            with count_lock:
                state["read_count"] -= 1
                if state["read_count"] == 0:
                    rw_lock.release()             # 最后一个读者放行写者
            time.sleep(random.uniform(0, 0.02))

    def writer(wid):
        for _ in range(2):
            with rw_lock:                          # 写者独占
                state["value"] += 1
                print(f"  写者{wid} 把 value 改为 {state['value']}")
                time.sleep(0.01)
            time.sleep(random.uniform(0, 0.02))

    ts = [threading.Thread(target=reader, args=(i,)) for i in range(3)]
    ts += [threading.Thread(target=writer, args=(i,)) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)
    print("  读者-写者结束，最终 value =", state["value"])


# ================================================= 3. 哲学家就餐
def dining_philosophers():
    print("\n===== 3. 哲学家就餐（5 人各吃 2 次，资源分级避免死锁）=====")
    N = 5
    forks = [threading.Lock() for _ in range(N)]
    eat_count = [0] * N

    def philosopher(i):
        left, right = i, (i + 1) % N
        # 关键：总是先拿编号小的叉子，破坏"循环等待"条件 → 不会死锁
        first, second = (left, right) if left < right else (right, left)
        for _ in range(2):
            with forks[first]:
                with forks[second]:
                    eat_count[i] += 1
                    print(f"  哲学家{i} 拿起叉子{first}和{second}，开吃（第{eat_count[i]}次）")
                    time.sleep(0.005)
            time.sleep(random.uniform(0, 0.01))   # 思考

    ts = [threading.Thread(target=philosopher, args=(i,)) for i in range(N)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)
    print("  就餐结束，各哲学家进食次数:", eat_count,
          "-> 无人饿死、无死锁" if all(c == 2 for c in eat_count) else "异常")


if __name__ == "__main__":
    producer_consumer()
    reader_writer()
    dining_philosophers()
    print("\n全部演示正常结束（若把哲学家改成'都先拿左手叉子'，就可能死锁挂住）")
