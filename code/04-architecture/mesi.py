# -*- coding: utf-8 -*-
"""MESI 缓存一致性协议状态机模拟。

两个核各有一个私有 cache，对同一地址执行读/写，
打印每一步操作后两个 cache 行的 MESI 状态变化。
状态：M(Modified) E(Exclusive) S(Shared) I(Invalid)
只用标准库。
"""

M, E, S, I = "M", "E", "S", "I"


class CacheLine(object):
    def __init__(self, name):
        self.name = name
        self.state = I

    def __repr__(self):
        return "{}:{}".format(self.name, self.state)


class Bus(object):
    """简化总线：广播读/写请求，其他 cache 侦听（snoop）并变更状态。"""

    def __init__(self, caches):
        self.caches = caches

    def read(self, requester):
        """requester 发起读。返回是否有其他 cache 持有该行（决定进 E 还是 S）。"""
        shared = False
        for c in self.caches:
            if c is requester:
                continue
            if c.state in (M, E, S):
                if c.state == M:
                    print("      [总线] {} 侦听到读请求, 写回脏数据".format(c.name))
                c.state = S          # 别人要读 -> 我降级为 Shared
                shared = True
        return shared

    def read_for_ownership(self, requester):
        """requester 发起写（RFO：Read For Ownership），其他副本全部失效。"""
        for c in self.caches:
            if c is requester:
                continue
            if c.state in (M, E, S):
                if c.state == M:
                    print("      [总线] {} 侦听到写请求, 写回脏数据".format(c.name))
                c.state = I          # 别人要写 -> 我失效


def cpu_read(cache, bus):
    if cache.state == I:                     # 读缺失
        shared = bus.read(cache)
        cache.state = S if shared else E     # 有别的副本进 S, 否则独占 E
    # M/E/S 状态下读命中，状态不变


def cpu_write(cache, bus):
    if cache.state in (I, S):                # 写缺失或需升级
        bus.read_for_ownership(cache)
    cache.state = M                          # 写完必为 Modified
    # E 状态写命中：静默升级为 M，不需要总线事务


def main():
    c0, c1 = CacheLine("Core0"), CacheLine("Core1")
    bus = Bus([c0, c1])

    script = [
        ("Core0 读 X", cpu_read, c0),
        ("Core0 写 X", cpu_write, c0),
        ("Core1 读 X", cpu_read, c1),
        ("Core1 写 X", cpu_write, c1),
        ("Core0 读 X", cpu_read, c0),
        ("Core0 读 X (命中)", cpu_read, c0),
    ]

    print("初始状态: {} {}".format(c0, c1))
    print("-" * 46)
    for desc, op, cache in script:
        op(cache, bus)
        print("{:<20} -> {} {}".format(desc, c0, c1))

    print("-" * 46)
    print("要点: 任意时刻最多一个 M; M/E 与其他有效副本互斥;")
    print("写共享行必须先让别人失效(RFO), 这就是伪共享开销的来源。")


if __name__ == "__main__":
    main()
