"""简化 Tomasulo 算法模拟器（仅用标准库）。

模型约定（教学简化，与 H&P 教材同构但参数可调）：
  1. 每拍最多发射（Issue）1 条指令，保留站满则停顿；
  2. 指令在发射的下一拍之后、且两个源操作数都已就绪时开始执行（Execute）；
  3. 执行占用 lat 拍，exec_end = exec_start + lat - 1；
  4. 写结果（Write Result）在 exec_end 的下一拍，通过 CDB 广播；
     每拍 CDB 只能广播一个结果，冲突时按发射顺序（老指令优先）；
  5. 广播后的下一拍，等待该结果的保留站才算操作数就绪（CDB 有一拍传播延迟）；
  6. 假设每个保留站背后有独立功能单元，不模拟功能单元结构冲突。

运行：python tomasulo.py
"""

# ---------------------------------------------------------------- 配置

LATENCY = {"LD": 2, "ADDD": 2, "SUBD": 2, "MULD": 6, "DIVD": 12}

# 保留站分组：名字前缀 -> (数量, 可执行的操作)
RS_GROUPS = [
    ("Load", 3, {"LD"}),
    ("Add", 3, {"ADDD", "SUBD"}),
    ("Mult", 2, {"MULD", "DIVD"}),
]

# 指令格式: (操作, 目的寄存器, 源1, 源2)  源为 None 表示立即数/地址
PROGRAM = [
    ("LD",   "F6",  None, None),   # LD   F6, 34(R2)
    ("LD",   "F2",  None, None),   # LD   F2, 45(R3)
    ("MULD", "F0",  "F2", "F4"),   # MULD F0, F2, F4
    ("SUBD", "F8",  "F6", "F2"),   # SUBD F8, F6, F2
    ("DIVD", "F10", "F0", "F6"),   # DIVD F10, F0, F6
    ("ADDD", "F6",  "F8", "F2"),   # ADDD F6, F8, F2
]

# 初始寄存器值（未被本段程序写入的寄存器）
INIT_REGS = {"F4": 3.0}


# ---------------------------------------------------------------- 数据结构

class RS(object):
    """保留站表项。字段与教材一致：Busy/Op/Vj/Vk/Qj/Qk。"""

    def __init__(self, name, ops):
        self.name = name
        self.ops = ops          # 本保留站支持的操作集合
        self.busy = False
        self.op = None
        self.vj = None          # 源操作数 1 的值（已就绪）
        self.vk = None          # 源操作数 2 的值
        self.qj = None          # 产生源 1 的保留站名（未就绪）
        self.qk = None          # 产生源 2 的保留站名
        self.idx = None         # 对应的指令序号（用于 CDB 仲裁与打印）
        self.exec_start = None
        self.exec_end = None

    def ready(self):
        return self.busy and self.qj is None and self.qk is None

    def cell(self):
        """一行紧凑文本，用于打印状态表。"""
        if not self.busy:
            return "{:<6} no   {:<5} {:>6} {:>6} {:>6} {:>6}".format(
                self.name, "-", "-", "-", "-", "-")
        fmt = lambda v: "-" if v is None else (
            "{:g}".format(v) if isinstance(v, float) else str(v))
        return "{:<6} yes  {:<5} {:>6} {:>6} {:>6} {:>6}".format(
            self.name, self.op, fmt(self.vj), fmt(self.vk),
            fmt(self.qj), fmt(self.qk))


def build_stations():
    stations = []
    for prefix, count, ops in RS_GROUPS:
        for i in range(1, count + 1):
            stations.append(RS("{}{}".format(prefix, i), ops))
    return stations


# ---------------------------------------------------------------- 模拟主体

def simulate(program, verbose=True, max_cycle=200):
    stations = build_stations()
    regs = dict(INIT_REGS)            # 寄存器的架构值
    qi = {}                           # 寄存器状态表: 寄存器 -> 正在写它的保留站名
    timing = [dict(issue=None, exec_start=None, exec_end=None, write=None)
              for _ in program]
    rs_of = [None] * len(program)     # 指令 -> 占用的保留站名
    pc = 0                            # 下一条待发射指令
    written = [False] * len(program)
    resolved_at = {}                  # 保留站名 -> 广播发生的拍号

    snapshots = []
    cycle = 0
    while cycle < max_cycle:
        cycle += 1
        event = []

        # ---- 阶段 1：Execute（用本拍开始时的状态判断就绪） ----
        for rs in stations:
            if not rs.busy or rs.exec_start is not None:
                continue
            if timing[rs.idx]["issue"] >= cycle:      # 发射当拍不能执行
                continue
            if not rs.ready():
                continue
            rs.exec_start = cycle
            rs.exec_end = cycle + LATENCY[rs.op] - 1
            timing[rs.idx]["exec_start"] = rs.exec_start
            timing[rs.idx]["exec_end"] = rs.exec_end
            event.append("{} 开始执行({}拍)".format(rs.name, LATENCY[rs.op]))

        # ---- 阶段 2：Write Result（CDB 每拍仅一个，老指令优先） ----
        done = [rs for rs in stations
                if rs.busy and rs.exec_end is not None and rs.exec_end < cycle]
        if done:
            rs = min(done, key=lambda r: r.idx)
            value = compute(rs)
            timing[rs.idx]["write"] = cycle
            written[rs.idx] = True
            resolved_at[rs.name] = cycle
            # 广播到所有等待它的保留站
            for other in stations:
                if other.qj == rs.name:
                    other.vj, other.qj = value, None
                if other.qk == rs.name:
                    other.vk, other.qk = value, None
            # 广播到寄存器状态表
            for r, owner in list(qi.items()):
                if owner == rs.name:
                    regs[r] = value
                    del qi[r]
            event.append("{} 经 CDB 广播 {:g}".format(rs.name, value))
            rs.busy = False
            rs.op = rs.vj = rs.vk = rs.qj = rs.qk = None
            rs.idx = rs.exec_start = rs.exec_end = None

        # ---- 阶段 3：Issue（每拍一条） ----
        if pc < len(program):
            op, dst, s1, s2 = program[pc]
            free = next((r for r in stations if not r.busy and op in r.ops), None)
            if free is not None:
                free.busy, free.op, free.idx = True, op, pc
                free.vj, free.vk = load_src(s1, regs, qi, free, "j"), None
                free.vk = load_src(s2, regs, qi, free, "k")
                qi[dst] = free.name            # 寄存器重命名：dst 归属该保留站
                timing[pc]["issue"] = cycle
                rs_of[pc] = free.name
                event.append("指令{} {} 发射到 {}".format(pc + 1, op, free.name))
                pc += 1
            else:
                event.append("保留站满，指令{} 停顿".format(pc + 1))

        snapshots.append((cycle, [rs.cell() for rs in stations],
                          dict(qi), "; ".join(event)))

        if pc >= len(program) and all(written):
            break

    if verbose:
        print_run(program, stations, snapshots, timing, rs_of)
    return timing, cycle


def load_src(src, regs, qi, rs, slot):
    """读取一个源操作数：已就绪返回值并写 V，未就绪写 Q 返回 None。"""
    if src is None:
        return 1.0                      # 立即数/地址，视为常量
    if src in qi:                       # 正被某个保留站计算
        setattr(rs, "q" + slot, qi[src])
        return None
    return regs.get(src, 1.0)


def compute(rs):
    """按操作类型算出结果值（数值本身不重要，用于演示 CDB 广播）。"""
    a = 1.0 if rs.vj is None else rs.vj
    b = 1.0 if rs.vk is None else rs.vk
    if rs.op == "LD":
        return 10.0                     # 假设从内存读回 10.0
    if rs.op == "ADDD":
        return a + b
    if rs.op == "SUBD":
        return a - b
    if rs.op == "MULD":
        return a * b
    if rs.op == "DIVD":
        return a / b if b != 0 else 0.0
    return 0.0


# ---------------------------------------------------------------- 打印

HEADER = "{:<6} {:<4} {:<5} {:>6} {:>6} {:>6} {:>6}".format(
    "Name", "Busy", "Op", "Vj", "Vk", "Qj", "Qk")


def asm(inst, n):
    op, dst, s1, s2 = inst
    src = ",".join(x for x in (s1, s2) if x) or "mem"
    return "{}. {:<5} {:<4} <- {}".format(n, op, dst, src)


def print_run(program, stations, snapshots, timing, rs_of, show=None):
    print("=" * 68)
    print("待模拟指令序列（延迟: " +
          ", ".join("{}={}".format(k, v) for k, v in sorted(LATENCY.items())) + "）")
    print("=" * 68)
    for i, inst in enumerate(program):
        print("  " + asm(inst, i + 1))

    show = show or [1, 2, 3, 4, 5, 6]
    for cycle, cells, qmap, event in snapshots:
        if cycle not in show:
            continue
        print("\n--- 第 {} 拍 --- {}".format(cycle, event))
        print("  " + HEADER)
        for c in cells:
            print("  " + c)
        print("  寄存器状态表 Qi: " +
              (", ".join("{}<-{}".format(k, v) for k, v in sorted(qmap.items()))
               or "(空)"))

    print("\n" + "=" * 68)
    print("各指令三阶段时刻表")
    print("=" * 68)
    print("  {:<22} {:>6} {:>10} {:>8}".format("指令", "Issue", "Execute", "Write"))
    for i, inst in enumerate(program):
        t = timing[i]
        print("  {:<22} {:>6} {:>10} {:>8}".format(
            asm(inst, i + 1), t["issue"],
            "{}-{}".format(t["exec_start"], t["exec_end"]), t["write"]))


def main():
    timing, total = simulate(PROGRAM, verbose=True)
    print("\n总耗时 {} 拍。".format(total))
    seq = sum(LATENCY[i[0]] for i in PROGRAM) + len(PROGRAM)
    print("若严格顺序执行（每条 发射1拍 + 执行lat拍）约需 {} 拍，".format(seq))
    print("Tomasulo 通过保留站 + CDB 让无关指令重叠，加速比约 {:.2f}x".format(
        seq / float(total)))


if __name__ == "__main__":
    main()
