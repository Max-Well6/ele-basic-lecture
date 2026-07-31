"""
Cache 模拟器：直接映射 / 组相联 / 全相联 + LRU / FIFO 替换
=========================================================
只用标准库。目标：用命中率数字说明「为什么循环换个顺序就快好几倍」。

地址划分（按字节地址）：
    | Tag | Set Index | Block Offset |
      高位     中间          低位
    block_offset_bits = log2(块大小)
    set_index_bits    = log2(组数)
    tag_bits          = 地址位数 - 上面两项

三种组织方式其实是同一个公式的特例：
    直接映射   = 1 路组相联（组数 = 行数）
    全相联     = 组数为 1 的组相联
    N 路组相联 = 中间态

运行：python cache_sim.py
"""

from collections import OrderedDict


class Cache:
    """通用组相联 Cache 模型（只跟踪命中/缺失，不存真实数据）。"""

    def __init__(self, size_bytes, block_bytes, ways, policy="LRU",
                 write_back=True, addr_bits=32):
        assert size_bytes % (block_bytes * ways) == 0, "容量必须能被 块大小×路数 整除"
        self.block_bytes = block_bytes
        self.ways = ways
        self.num_sets = size_bytes // (block_bytes * ways)
        self.policy = policy
        self.write_back = write_back
        self.addr_bits = addr_bits

        # 位宽计算
        self.offset_bits = self.num_sets and block_bytes.bit_length() - 1
        self.index_bits = self.num_sets.bit_length() - 1
        self.tag_bits = addr_bits - self.offset_bits - self.index_bits

        # 每组一个 OrderedDict: tag -> dirty标志。顺序即替换优先级
        self.sets = [OrderedDict() for _ in range(self.num_sets)]

        self.hits = self.misses = 0
        self.writebacks = 0
        self.compulsory = 0      # 冷启动缺失（该块从未出现过）
        self._seen_blocks = set()

    # ---- 地址拆解 ----
    def split(self, addr):
        block_addr = addr >> self.offset_bits
        index = block_addr & (self.num_sets - 1)
        tag = block_addr >> self.index_bits
        return tag, index

    def access(self, addr, is_write=False):
        """访问一次内存地址，返回 True 表示命中。"""
        tag, index = self.split(addr)
        s = self.sets[index]
        block_addr = addr >> self.offset_bits

        if tag in s:
            self.hits += 1
            if self.policy == "LRU":
                s.move_to_end(tag)          # LRU：命中后挪到最新端
            if is_write and self.write_back:
                s[tag] = True               # 标记脏位
            return True

        # ---- 缺失处理 ----
        self.misses += 1
        if block_addr not in self._seen_blocks:
            self.compulsory += 1
            self._seen_blocks.add(block_addr)

        if len(s) >= self.ways:
            # 替换：OrderedDict 的头部就是 LRU / FIFO 的牺牲者
            victim_tag, dirty = s.popitem(last=False)
            if dirty:
                self.writebacks += 1        # 写回策略下脏块要刷回内存
        s[tag] = bool(is_write and self.write_back)
        return False

    # ---- 统计 ----
    @property
    def total(self):
        return self.hits + self.misses

    @property
    def hit_rate(self):
        return self.hits / self.total if self.total else 0.0

    def amat(self, hit_time=1.0, miss_penalty=100.0):
        """平均访存时间 AMAT = 命中时间 + 缺失率 × 缺失代价。"""
        return hit_time + (1 - self.hit_rate) * miss_penalty

    def config_str(self):
        kind = ("直接映射" if self.ways == 1 else
                "全相联" if self.num_sets == 1 else f"{self.ways}路组相联")
        size = self.num_sets * self.ways * self.block_bytes
        return (f"{size}B/{self.block_bytes}B块/{kind}/{self.policy} "
                f"[tag={self.tag_bits} index={self.index_bits} offset={self.offset_bits}]")

    def summary(self):
        return (f"访问 {self.total:>6}  命中 {self.hits:>6}  缺失 {self.misses:>5}  "
                f"命中率 {self.hit_rate * 100:>6.2f}%  AMAT {self.amat():>6.2f}周期")


# --------------------------------------------------------------------------
# 访存序列生成器：模拟典型程序的地址流
# --------------------------------------------------------------------------
def seq_stream(n, stride=4, base=0):
    """顺序/跨步访问：stride=4 表示逐个 int 遍历数组。"""
    return [base + i * stride for i in range(n)]


def matmul_stream(n, order="ijk", elem=4, base_a=0, base_b=None, base_c=None):
    """生成 n×n 矩阵乘法的访存地址流，对比不同循环顺序的局部性。

    C[i][j] += A[i][k] * B[k][j]
    ijk 顺序：B 按列访问，跨步 n*elem，局部性差
    ikj 顺序：B 按行访问，跨步 elem，局部性好
    """
    span = n * n * elem
    base_b = span if base_b is None else base_b
    base_c = 2 * span if base_c is None else base_c
    A = lambda i, k: base_a + (i * n + k) * elem
    B = lambda k, j: base_b + (k * n + j) * elem
    C = lambda i, j: base_c + (i * n + j) * elem

    stream = []
    if order == "ijk":
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    stream += [(A(i, k), False), (B(k, j), False),
                               (C(i, j), False), (C(i, j), True)]
    elif order == "ikj":
        for i in range(n):
            for k in range(n):
                for j in range(n):
                    stream += [(A(i, k), False), (B(k, j), False),
                               (C(i, j), False), (C(i, j), True)]
    else:
        raise ValueError(order)
    return stream


def loop_tiled_stream(n, tile, elem=4):
    """分块（tiling）后的矩阵乘法地址流：AI 算子优化的基本手法。"""
    span = n * n * elem
    A = lambda i, k: (i * n + k) * elem
    B = lambda k, j: span + (k * n + j) * elem
    C = lambda i, j: 2 * span + (i * n + j) * elem
    stream = []
    for ii in range(0, n, tile):
        for kk in range(0, n, tile):
            for jj in range(0, n, tile):
                for i in range(ii, min(ii + tile, n)):
                    for k in range(kk, min(kk + tile, n)):
                        for j in range(jj, min(jj + tile, n)):
                            stream += [(A(i, k), False), (B(k, j), False),
                                       (C(i, j), False), (C(i, j), True)]
    return stream


def run(cache, stream):
    """stream 可以是纯地址列表，也可以是 (addr, is_write) 元组列表。"""
    for item in stream:
        if isinstance(item, tuple):
            cache.access(item[0], item[1])
        else:
            cache.access(item)
    return cache


# --------------------------------------------------------------------------
# 实验
# --------------------------------------------------------------------------
def exp_mapping():
    print("=" * 78)
    print("实验 1：同样 1KB 容量，映射方式对命中率的影响（顺序遍历 2048 个 int）")
    print("=" * 78)
    stream = seq_stream(2048, stride=4)
    for ways in (1, 2, 4, 8, 16):
        c = Cache(1024, 32, ways)
        run(c, stream)
        print(f"{c.config_str():<58}{c.hit_rate * 100:>6.2f}%")
    print("\n结论: 纯顺序访问下相联度几乎无用——缺失全是「冷启动缺失」，")
    print("      一个块 32B 装 8 个 int，所以命中率天然趋近 7/8 = 87.5%。")


def exp_block_size():
    print("\n" + "=" * 78)
    print("实验 2：块大小的影响（同为 1KB、4 路组相联）")
    print("=" * 78)
    stream = seq_stream(2048, stride=4)
    for bs in (8, 16, 32, 64, 128):
        c = Cache(1024, bs, 4)
        run(c, stream)
        print(f"块大小 {bs:>4}B  命中率 {c.hit_rate * 100:>6.2f}%  "
              f"缺失 {c.misses:>5}  强制缺失 {c.compulsory:>5}")
    print("\n结论: 块越大，空间局部性利用越充分，但块太大会挤走有用数据（污染），")
    print("      且缺失代价上升。真实 CPU 普遍取 64B，就是这条 U 形曲线的谷底。")


def exp_conflict():
    print("\n" + "=" * 78)
    print("实验 3：冲突缺失——直接映射的痛点（跨步访问，步长恰好等于缓存大小）")
    print("=" * 78)
    # 步长 1024B 恰好等于 1KB 缓存容量，4 个地址全部映射到同一组
    bad = [i % 4 * 1024 for i in range(4096)]
    for ways in (1, 2, 4, 8):
        c = Cache(1024, 32, ways)
        run(c, bad)
        print(f"{c.config_str():<58}{c.hit_rate * 100:>6.2f}%")
    print("\n结论: 只有 4 个地址在轮流访问，容量绰绰有余，直接映射却 0% 命中——")
    print("      因为它们争抢同一个组。路数 >= 冲突地址数（这里是 4）时问题消失。")
    print("      这就是「缓存抖动 thrashing」，数组维度取 2 的幂时特别容易踩到。")


def exp_replacement():
    print("\n" + "=" * 78)
    print("实验 4：LRU vs FIFO")
    print("=" * 78)

    # 场景 A：热点 + 流式扫描。热点块被反复访问，FIFO 却会把它按「入队顺序」踢掉
    hot = 0
    stream_a = []
    for k in range(1, 400):
        stream_a.append(hot)                 # 热点数据，理应常驻
        stream_a.append((k % 64 + 1) * 4096)  # 流式数据，与热点映射到同一组
    print("场景A 热点数据 + 流式扫描（映射到同一组）:")
    for policy in ("LRU", "FIFO"):
        c = Cache(1024, 32, 8, policy=policy)
        run(c, stream_a)
        print(f"  {policy:<6} {c.summary()}")

    # 场景 B：工作集略大于容量的循环访问，两者一起失效
    stream_b = [(i % 40) * 32 for _ in range(20) for i in range(40)]
    print("场景B 循环访问 40 块 > 32 块容量:")
    for policy in ("LRU", "FIFO"):
        c = Cache(1024, 32, 8, policy=policy)
        run(c, stream_b)
        print(f"  {policy:<6} {c.summary()}")

    print("\n结论: 场景A 说明 LRU 能保住热点数据，FIFO 不看使用频度所以吃亏；")
    print("      场景B 说明 LRU 也不是万能的——工作集稍大于容量时循环访问全灭，")
    print("      称为「顺序抖动」。真实处理器用伪 LRU / RRIP 兼顾硬件开销与抗抖动。")


def exp_matmul():
    print("\n" + "=" * 78)
    print("实验 5：矩阵乘法循环顺序 —— 一行代码顺序换来数倍性能")
    print("=" * 78)
    n = 48
    for order in ("ijk", "ikj"):
        c = Cache(4096, 64, 4)
        run(c, matmul_stream(n, order))
        print(f"{n}x{n} {order} 顺序  {c.summary()}")
    c = Cache(4096, 64, 4)
    run(c, loop_tiled_stream(n, 8))
    print(f"{n}x{n} 分块(tile=8)  {c.summary()}")
    print("\n结论: ikj 让 B 矩阵按行连续访问，命中率大幅提升；分块进一步把工作集")
    print("      压进 Cache。这正是 BLAS / cuBLAS / AI 编译器做算子优化的核心思路。")


def exp_amat():
    print("\n" + "=" * 78)
    print("实验 6：命中率的微小变化如何放大成性能鸿沟（缺失代价 200 周期）")
    print("=" * 78)
    print(f"{'命中率':>8}{'缺失率':>10}{'AMAT(周期)':>14}{'相对 99% 的倍数':>18}")
    base = None
    for hr in (0.99, 0.97, 0.95, 0.90, 0.80, 0.50):
        amat = 1 + (1 - hr) * 200
        base = base or amat
        print(f"{hr * 100:>7.0f}%{(1 - hr) * 100:>9.0f}%{amat:>14.2f}{amat / base:>18.2f}x")
    print("\n结论: 命中率从 99% 掉到 95%，AMAT 就翻了 4 倍。")
    print("      这就是「内存墙」：不是内存慢了，是 CPU 快得等不起。")


if __name__ == "__main__":
    exp_mapping()
    exp_block_size()
    exp_conflict()
    exp_replacement()
    exp_matmul()
    exp_amat()
    print("\n全部实验完成。")
