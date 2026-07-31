"""
五级流水线时空图模拟器：冒险检测、转发与分支惩罚
================================================
只用标准库。目标：把课本上那张「IF ID EX MEM WB 阶梯图」自动画出来，
并用同一段代码定量比较「有无转发」「有无分支预测」的 CPI 差距。

五级流水（经典 MIPS/RISC-V 教学模型）：
    IF  取指      从指令存储器读指令，PC+4
    ID  译码      解析指令、读寄存器堆、生成控制信号
    EX  执行      ALU 运算 / 计算访存地址 / 判断分支
    MEM 访存      只有 load/store 真正用到
    WB  写回      把结果写进寄存器堆

三类冒险：
    结构冒险  硬件资源冲突（本模型假设指令/数据存储器分离，故不发生）
    数据冒险  RAW（写后读）是流水线里唯一真正棘手的一种
    控制冒险  分支要到 EX 才知道跳不跳，其后已经取进来的指令得作废

运行：python pipeline_viz.py
"""

STAGES = ["IF", "ID", "EX", "ME", "WB"]

# 指令类型：决定它读哪些寄存器、写哪个寄存器、结果何时就绪
#   kind: "alu" 结果在 EX 末尾就绪；"load" 要等到 MEM 末尾；
#         "store"/"branch" 不写寄存器
KINDS = {"ADD", "SUB", "AND", "OR", "SLT", "ADDI", "LOAD", "STORE", "BEQ", "BNE", "NOP"}


class Instr:
    def __init__(self, text):
        self.text = text.strip()
        parts = self.text.replace(",", " ").split()
        self.op = parts[0].upper()
        assert self.op in KINDS, f"未知指令 {self.op}"
        regs = [p for p in parts[1:] if p.upper().startswith("R")]

        if self.op in ("LOAD",):
            self.rd, self.srcs, self.kind = regs[0], regs[1:], "load"
        elif self.op in ("STORE",):
            self.rd, self.srcs, self.kind = None, regs, "store"
        elif self.op in ("BEQ", "BNE"):
            self.rd, self.srcs, self.kind = None, regs, "branch"
        elif self.op == "NOP":
            self.rd, self.srcs, self.kind = None, [], "alu"
        elif self.op == "ADDI":
            self.rd, self.srcs, self.kind = regs[0], regs[1:2], "alu"
        else:
            self.rd, self.srcs, self.kind = regs[0], regs[1:], "alu"

    def __str__(self):
        return self.text


def simulate(program, forwarding=True, branch_taken=None, branch_penalty=2):
    """模拟流水线，返回每条指令的阶段时刻表与统计信息。

    branch_taken: 集合，包含哪些指令下标是「实际发生跳转」的分支。
                  发生跳转时后续指令要被冲刷，代价 branch_penalty 个周期。
    """
    insts = [Instr(t) for t in program]
    branch_taken = branch_taken or set()

    sched = []      # 每项: dict(if, id, ex, me, wb, stall, bubble_before)
    prev = None
    data_stalls = ctrl_stalls = 0

    for i, ins in enumerate(insts):
        # ---- IF：默认紧跟上一条，遇到已跳转的分支要加冲刷惩罚 ----
        if prev is None:
            if_c = 1
        else:
            if_c = prev["if"] + 1
            if (i - 1) in branch_taken:
                if_c += branch_penalty
                ctrl_stalls += branch_penalty

        id_c = if_c + 1
        if prev:
            id_c = max(id_c, prev["id"] + 1)

        # ---- 数据冒险：找出所有仍在流水线中的生产者 ----
        ex_c = id_c + 1
        hazard_note = []
        for j in range(i - 1, max(-1, i - 5), -1):
            prod, ps = insts[j], sched[j]
            if not prod.rd or prod.rd not in ins.srcs:
                continue
            if forwarding:
                if prod.kind == "load":
                    # load-use 冒险：数据要等 MEM 末尾，转发也救不了，必须停 1 拍
                    need = ps["me"] + 1
                    if need > ex_c:
                        hazard_note.append(f"load-use({prod.rd})")
                    ex_c = max(ex_c, need)
                else:
                    # EX->EX / MEM->EX 转发：结果一出炉就直接送回 ALU 输入
                    ex_c = max(ex_c, ps["ex"] + 1)
            else:
                # 无转发：必须等生产者写回寄存器堆（假设写在前半拍、读在后半拍）
                need = ps["wb"] + 1
                if need > ex_c:
                    hazard_note.append(f"RAW({prod.rd})")
                ex_c = max(ex_c, need)
            break  # 只需最近的那个生产者

        if prev:
            ex_c = max(ex_c, prev["ex"] + 1)   # 保持顺序，不能超车

        stall = ex_c - (id_c + 1)
        data_stalls += stall

        rec = {"if": if_c, "id": id_c, "ex": ex_c, "me": ex_c + 1, "wb": ex_c + 2,
               "stall": stall, "note": ",".join(hazard_note), "ins": ins}
        sched.append(rec)
        prev = rec

    total_cycles = sched[-1]["wb"]
    return {
        "sched": sched, "cycles": total_cycles, "n": len(insts),
        "cpi": total_cycles / len(insts),
        "data_stalls": data_stalls, "ctrl_stalls": ctrl_stalls,
    }


def draw(result, title=""):
    """打印 ASCII 时空图：横轴时钟周期，纵轴指令。"""
    sched = result["sched"]
    total = result["cycles"]
    width = 3
    if title:
        print(f"\n{title}")
    header = " " * 22 + "".join(f"{c:>{width}}" for c in range(1, total + 1))
    print(header)
    print(" " * 22 + "-" * (width * total))
    for r in sched:
        row = [""] * (total + 1)
        row[r["if"]] = "IF"
        row[r["id"]] = "ID"
        # 停顿期间指令卡在 ID，用 * 表示气泡
        for c in range(r["id"] + 1, r["ex"]):
            row[c] = " *"
        row[r["ex"]] = "EX"
        row[r["me"]] = "ME"
        row[r["wb"]] = "WB"
        line = "".join(f"{row[c]:>{width}}" for c in range(1, total + 1))
        note = f"  <- {r['note']}" if r["note"] else ""
        print(f"{str(r['ins']):<22}{line}{note}")
    print(f"\n  指令数={result['n']}  总周期={result['cycles']}  "
          f"CPI={result['cpi']:.2f}  数据停顿={result['data_stalls']}  "
          f"控制停顿={result['ctrl_stalls']}")


# --------------------------------------------------------------------------
# 示例程序
# --------------------------------------------------------------------------
DEP_CHAIN = [
    "ADD  R1, R2, R3",
    "SUB  R4, R1, R5",     # 依赖 R1，紧邻上一条
    "AND  R6, R1, R4",     # 同时依赖 R1 和 R4
    "OR   R7, R6, R2",
]

LOAD_USE = [
    "LOAD R1, R9",
    "ADD  R2, R1, R3",     # load-use：转发也必须停 1 拍
    "SUB  R4, R2, R5",
    "STORE R4, R9",
]

LOAD_USE_FIXED = [
    "LOAD R1, R9",
    "OR   R8, R6, R7",     # 编译器把一条无关指令调度到这里，填掉气泡
    "ADD  R2, R1, R3",
    "SUB  R4, R2, R5",
]

BRANCH_PROG = [
    "ADD  R1, R2, R3",
    "BEQ  R1, R4",         # 假设跳转发生
    "AND  R5, R6, R7",     # 会被冲刷
    "OR   R8, R9, R1",     # 会被冲刷
    "ADD  R2, R2, R1",
]


def exp_forwarding():
    print("=" * 78)
    print("实验 1：数据冒险 —— 转发（Forwarding）值多少钱")
    print("=" * 78)
    a = simulate(DEP_CHAIN, forwarding=False)
    draw(a, "【无转发】必须等生产者 WB 完成，寄存器堆是唯一的通路")
    b = simulate(DEP_CHAIN, forwarding=True)
    draw(b, "【有转发】ALU 结果一出来就抄近道送回 ALU 输入端")
    print(f"\n  加速比 = {a['cycles']} / {b['cycles']} = {a['cycles'] / b['cycles']:.2f}x")
    print("  结论: 连续依赖的代码里，转发几乎消灭了全部数据停顿。")


def exp_load_use():
    print("\n" + "=" * 78)
    print("实验 2：load-use 冒险 —— 转发也救不了的一拍")
    print("=" * 78)
    a = simulate(LOAD_USE, forwarding=True)
    draw(a, "【转发已开启】LOAD 的数据 MEM 末尾才到，下一条的 EX 只能等")
    b = simulate(LOAD_USE_FIXED, forwarding=True)
    draw(b, "【编译器调度后】插入一条无关指令填气泡，停顿消失")
    print(f"\n  周期 {a['cycles']} -> {b['cycles']}，停顿 {a['data_stalls']} -> {b['data_stalls']}")
    print("  结论: 这是编译器指令调度（instruction scheduling）最基本的价值。")


def exp_branch():
    print("\n" + "=" * 78)
    print("实验 3：控制冒险 —— 分支惩罚")
    print("=" * 78)
    a = simulate(BRANCH_PROG, forwarding=True, branch_taken={1}, branch_penalty=2)
    draw(a, "【分支在 EX 判定，跳转发生】后面 2 条已取指的指令作废")
    b = simulate(BRANCH_PROG, forwarding=True, branch_taken=set())
    draw(b, "【预测正确 / 分支不跳转】零惩罚")
    print(f"\n  预测失败代价: {a['cycles'] - b['cycles']} 个周期")


def exp_cpi_model():
    print("\n" + "=" * 78)
    print("实验 4：分支预测准确率如何决定实际 CPI")
    print("=" * 78)
    branch_freq = 0.20      # 分支指令占比
    penalty = 15            # 现代深流水线的预测失败代价（周期）
    print(f"  假设：分支占比 {branch_freq:.0%}，预测失败惩罚 {penalty} 周期，理想 CPI = 1")
    print(f"\n{'预测准确率':>12}{'失败率':>10}{'附加CPI':>12}{'实际CPI':>12}{'相对性能':>12}")
    base = None
    for acc in (0.99, 0.98, 0.95, 0.90, 0.80, 0.50):
        extra = branch_freq * (1 - acc) * penalty
        cpi = 1 + extra
        base = base or cpi
        print(f"{acc:>11.0%}{1 - acc:>10.0%}{extra:>12.3f}{cpi:>12.3f}{base / cpi:>11.2f}x")
    print("\n  结论: 流水线越深、分支惩罚越大，预测器就越关键。")
    print("        这就是现代 CPU 拿出上百 KB 晶体管做分支预测器的原因。")


def exp_speedup():
    print("\n" + "=" * 78)
    print("实验 5：流水线的理论加速比与现实差距")
    print("=" * 78)
    k, delays = 5, [2.0, 1.0, 2.0, 2.0, 1.0]
    t_seq = sum(delays)
    t_pipe = max(delays) + 0.2      # 加上流水线寄存器的建立时间
    print(f"  非流水线时钟周期 = {t_seq}ns，流水线时钟周期 = {t_pipe}ns")
    print(f"\n{'指令数 n':>10}{'非流水(ns)':>14}{'流水(ns)':>13}{'加速比':>10}{'效率':>10}")
    for n in (1, 5, 10, 100, 1000, 100000):
        t1 = n * t_seq
        t2 = (k + n - 1) * t_pipe
        print(f"{n:>10}{t1:>14.1f}{t2:>13.1f}{t1 / t2:>9.2f}x{t1 / t2 / k:>9.1%}")
    print(f"\n  理论上限 = 各段延迟不均衡时为 {t_seq / t_pipe:.2f}x（不是级数 5）")
    print("  结论: 加速比受限于最慢一级，且流水线寄存器有固定开销；")
    print("        「切得越细越快」在 Pentium 4 的 31 级流水线上被证伪。")


if __name__ == "__main__":
    exp_forwarding()
    exp_load_use()
    exp_branch()
    exp_cpi_model()
    exp_speedup()
    print("\n全部实验完成。")
