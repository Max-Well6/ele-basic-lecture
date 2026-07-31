"""
TOY-16：一个 16 位玩具 ISA 的单周期 CPU 模拟器（含两趟汇编器）
==============================================================
只用标准库。目标：把「取指-译码-执行-访存-写回」从课本插图变成能跑的代码。

------------------------------------------------------------------
一、体系结构规格（ISA）
------------------------------------------------------------------
* 字长        : 16 位，寄存器与内存单元均为 16 位有符号数（补码）
* 寄存器      : R0~R7 共 8 个；R0 恒为 0（写入被丢弃，仿 RISC-V 的 x0）
* 指令存储器  : 最多 512 条指令，按字寻址，PC 单位是「条」
* 数据存储器  : 256 个字，地址 0~255
* 指令长度    : 定长 16 位（RISC 风格，译码简单）

指令格式（三种）：
    R 型:  op[15:12] | rd[11:9] | rs1[8:6] | rs2[5:3] | 000[2:0]
    I 型:  op[15:12] | rd[11:9] | imm[8:0]        (imm 为 9 位补码, -256~255)
    M/J型: op[15:12] | rd[11:9] | addr[8:0]       (addr 为 9 位无符号)

指令集（13 条，够写循环和数组处理）：
    0x0 HALT                停机
    0x1 LOADI rd, imm       rd <- imm
    0x2 LOAD  rd, addr      rd <- DMEM[addr]
    0x3 STORE rs, addr      DMEM[addr] <- rs
    0x4 ADD   rd, rs1, rs2  rd <- rs1 + rs2
    0x5 SUB   rd, rs1, rs2  rd <- rs1 - rs2
    0x6 AND   rd, rs1, rs2  rd <- rs1 & rs2
    0x7 OR    rd, rs1, rs2  rd <- rs1 | rs2
    0x8 SLT   rd, rs1, rs2  rd <- (rs1 < rs2) ? 1 : 0
    0x9 ADDI  rd, imm       rd <- rd + imm
    0xA JMP   addr          PC <- addr
    0xB JZ    rs, addr      if rs == 0 then PC <- addr
    0xC JNZ   rs, addr      if rs != 0 then PC <- addr
    0xD OUT   rs            把 rs 的值送到输出端口（打印）

运行：python toy_cpu.py
"""

WORD_BITS = 16
WORD_MASK = (1 << WORD_BITS) - 1
DMEM_SIZE = 256
NUM_REGS = 8


# --------------------------------------------------------------------------
# 补码工具：模拟器内部用 Python 无界整数，进出边界时做位宽截断
# --------------------------------------------------------------------------
def to_signed(x: int, bits: int = WORD_BITS) -> int:
    """把 bits 位的位模式解释为补码有符号数。"""
    x &= (1 << bits) - 1
    return x - (1 << bits) if x >> (bits - 1) else x


def to_unsigned(x: int, bits: int = WORD_BITS) -> int:
    """把有符号数截断成 bits 位的位模式。"""
    return x & ((1 << bits) - 1)


# --------------------------------------------------------------------------
# 指令集定义表：一处定义，汇编器和译码器共用，避免两边不一致
# --------------------------------------------------------------------------
OPCODES = {
    "HALT": 0x0, "LOADI": 0x1, "LOAD": 0x2, "STORE": 0x3,
    "ADD": 0x4, "SUB": 0x5, "AND": 0x6, "OR": 0x7, "SLT": 0x8,
    "ADDI": 0x9, "JMP": 0xA, "JZ": 0xB, "JNZ": 0xC, "OUT": 0xD,
}
OPNAMES = {v: k for k, v in OPCODES.items()}

# 每条指令的操作数形态，汇编器据此解析
#   "R"   : rd, rs1, rs2      "I": rd, imm       "M": reg, addr
#   "J"   : addr              "S": reg           "N": 无操作数
FORMS = {
    "HALT": "N", "LOADI": "I", "LOAD": "M", "STORE": "M",
    "ADD": "R", "SUB": "R", "AND": "R", "OR": "R", "SLT": "R",
    "ADDI": "I", "JMP": "J", "JZ": "M", "JNZ": "M", "OUT": "S",
}


# --------------------------------------------------------------------------
# 汇编器：两趟扫描。第一趟建立标签->地址映射，第二趟生成机器码。
# --------------------------------------------------------------------------
def _parse_reg(tok: str) -> int:
    tok = tok.strip().upper()
    if not tok.startswith("R") or not tok[1:].isdigit():
        raise ValueError(f"非法寄存器: {tok}")
    n = int(tok[1:])
    if not 0 <= n < NUM_REGS:
        raise ValueError(f"寄存器越界: {tok}")
    return n


def _clean(line: str) -> str:
    """去掉注释（; 或 #）和首尾空白。"""
    for mark in (";", "#"):
        idx = line.find(mark)
        if idx >= 0:
            line = line[:idx]
    return line.strip()


def assemble(source: str) -> list:
    """把汇编文本翻译成机器码列表（每项一个 16 位整数）。"""
    raw_lines = [_clean(ln) for ln in source.splitlines()]

    # ---- 第一趟：确定每条指令的地址，收集标签 ----
    labels, pc = {}, 0
    body = []
    for ln in raw_lines:
        if not ln:
            continue
        while ":" in ln:  # 支持 "loop: ADD ..." 和独占一行的标签
            label, _, rest = ln.partition(":")
            labels[label.strip()] = pc
            ln = rest.strip()
            if not ln:
                break
        if ln:
            body.append((pc, ln))
            pc += 1

    # ---- 第二趟：编码 ----
    code = []
    for addr, ln in body:
        parts = ln.replace(",", " ").split()
        mnem = parts[0].upper()
        if mnem not in OPCODES:
            raise ValueError(f"未知指令: {mnem} (行: {ln})")
        op, form, args = OPCODES[mnem], FORMS[mnem], parts[1:]

        def resolve(tok: str) -> int:
            """立即数/地址可以是数字，也可以是标签。"""
            tok = tok.strip()
            if tok in labels:
                return labels[tok]
            return int(tok, 0)

        if form == "N":
            word = op << 12
        elif form == "R":
            rd, rs1, rs2 = (_parse_reg(a) for a in args[:3])
            word = (op << 12) | (rd << 9) | (rs1 << 6) | (rs2 << 3)
        elif form == "I":
            rd, imm = _parse_reg(args[0]), resolve(args[1])
            if not -256 <= imm <= 255:
                raise ValueError(f"立即数超出 9 位补码范围: {imm}")
            word = (op << 12) | (rd << 9) | (imm & 0x1FF)
        elif form == "M":
            reg, a = _parse_reg(args[0]), resolve(args[1])
            word = (op << 12) | (reg << 9) | (a & 0x1FF)
        elif form == "J":
            word = (op << 12) | (resolve(args[0]) & 0x1FF)
        elif form == "S":
            word = (op << 12) | (_parse_reg(args[0]) << 9)
        else:
            raise AssertionError(form)
        code.append(word & WORD_MASK)
    return code


def disassemble(word: int) -> str:
    """反汇编单条指令，供 trace 输出使用。"""
    op = (word >> 12) & 0xF
    rd = (word >> 9) & 0x7
    rs1 = (word >> 6) & 0x7
    rs2 = (word >> 3) & 0x7
    imm = to_signed(word & 0x1FF, 9)
    addr = word & 0x1FF
    name = OPNAMES.get(op, "???")
    form = FORMS.get(name, "N")
    if form == "N":
        return name
    if form == "R":
        return f"{name} R{rd}, R{rs1}, R{rs2}"
    if form == "I":
        return f"{name} R{rd}, {imm}"
    if form == "M":
        return f"{name} R{rd}, [{addr}]"
    if form == "J":
        return f"{name} {addr}"
    return f"{name} R{rd}"


# --------------------------------------------------------------------------
# CPU 本体：严格按单周期数据通路的五个功能段组织
# --------------------------------------------------------------------------
class ToyCPU:
    """单周期实现：一条指令在一个时钟周期内走完全部五段，因此 CPI = 1。

    代价是时钟周期必须容纳最慢的那条指令（LOAD：取指+译码+ALU+访存+写回），
    这正是后面引出多周期与流水线的动机。
    """

    # 各功能段的组合逻辑延迟（ns），用于估算单周期时钟周期
    STAGE_DELAY = {"IF": 2.0, "ID": 1.0, "EX": 2.0, "MEM": 2.0, "WB": 1.0}

    def __init__(self, code, trace=False):
        self.imem = list(code)
        self.dmem = [0] * DMEM_SIZE
        self.reg = [0] * NUM_REGS
        self.pc = 0
        self.halted = False
        self.cycles = 0          # 单周期机中，周期数 == 指令数
        self.instr_count = 0
        self.trace = trace
        self.output = []
        self.op_stat = {}        # 指令类型直方图，用于算加权 CPI

    # ---- 寄存器读写（封装 R0 恒零的硬件特性）----
    def _read(self, r: int) -> int:
        return 0 if r == 0 else self.reg[r]

    def _write(self, r: int, v: int) -> None:
        if r != 0:
            self.reg[r] = to_signed(to_unsigned(v))  # 溢出即回绕，符合硬件行为

    # ---- 五个功能段 ----
    def _fetch(self):
        """IF：按 PC 取指，PC+1 指向下一条。"""
        if self.pc >= len(self.imem):
            self.halted = True
            return None
        word = self.imem[self.pc]
        self.pc += 1
        return word

    @staticmethod
    def _decode(word: int) -> dict:
        """ID：把 16 位指令字拆成控制信号和操作数编号。"""
        return {
            "op": (word >> 12) & 0xF,
            "rd": (word >> 9) & 0x7,
            "rs1": (word >> 6) & 0x7,
            "rs2": (word >> 3) & 0x7,
            "imm": to_signed(word & 0x1FF, 9),
            "addr": word & 0x1FF,
            "word": word,
        }

    def _execute(self, d: dict):
        """EX + MEM + WB：单周期机里这三段在同一周期内串行完成。"""
        op = d["op"]
        name = OPNAMES.get(op, "???")
        self.op_stat[name] = self.op_stat.get(name, 0) + 1

        if op == OPCODES["HALT"]:
            self.halted = True
        elif op == OPCODES["LOADI"]:
            self._write(d["rd"], d["imm"])
        elif op == OPCODES["LOAD"]:                      # MEM 段读
            self._write(d["rd"], self.dmem[d["addr"] % DMEM_SIZE])
        elif op == OPCODES["STORE"]:                     # MEM 段写
            self.dmem[d["addr"] % DMEM_SIZE] = self._read(d["rd"])
        elif op == OPCODES["ADD"]:
            self._write(d["rd"], self._read(d["rs1"]) + self._read(d["rs2"]))
        elif op == OPCODES["SUB"]:
            self._write(d["rd"], self._read(d["rs1"]) - self._read(d["rs2"]))
        elif op == OPCODES["AND"]:
            self._write(d["rd"], to_unsigned(self._read(d["rs1"])) & to_unsigned(self._read(d["rs2"])))
        elif op == OPCODES["OR"]:
            self._write(d["rd"], to_unsigned(self._read(d["rs1"])) | to_unsigned(self._read(d["rs2"])))
        elif op == OPCODES["SLT"]:
            self._write(d["rd"], 1 if self._read(d["rs1"]) < self._read(d["rs2"]) else 0)
        elif op == OPCODES["ADDI"]:
            self._write(d["rd"], self._read(d["rd"]) + d["imm"])
        elif op == OPCODES["JMP"]:
            self.pc = d["addr"]
        elif op == OPCODES["JZ"]:
            if self._read(d["rd"]) == 0:
                self.pc = d["addr"]
        elif op == OPCODES["JNZ"]:
            if self._read(d["rd"]) != 0:
                self.pc = d["addr"]
        elif op == OPCODES["OUT"]:
            self.output.append(self._read(d["rd"]))
        else:
            raise ValueError(f"非法操作码 0x{op:X} @PC={self.pc - 1}")

    def step(self):
        """执行一个时钟周期 = 一条完整指令。"""
        pc_before = self.pc
        word = self._fetch()
        if word is None:
            return
        d = self._decode(word)
        self._execute(d)
        self.cycles += 1
        self.instr_count += 1
        if self.trace:
            regs = " ".join(f"R{i}={self._read(i):<5}" for i in range(1, 5))
            print(f"  cyc{self.cycles:>3} PC={pc_before:<3} "
                  f"{disassemble(word):<20} | {regs}")

    def run(self, max_cycles: int = 100000):
        while not self.halted and self.cycles < max_cycles:
            self.step()
        if self.cycles >= max_cycles:
            raise RuntimeError("超过最大周期数，可能死循环")
        return self

    # ---- 性能模型 ----
    def clock_period_ns(self) -> float:
        """单周期机的时钟周期 = 最长路径（LOAD 指令要走满五段）。"""
        return sum(self.STAGE_DELAY.values())

    def report(self):
        t = self.clock_period_ns()
        total = self.cycles * t
        print(f"\n[性能报告] 指令数={self.instr_count}  周期数={self.cycles}  CPI={self.cycles / self.instr_count:.2f}")
        print(f"           时钟周期={t:.1f}ns (f={1000 / t:.0f}MHz)  程序执行时间={total:.1f}ns")
        hist = ", ".join(f"{k}x{v}" for k, v in sorted(self.op_stat.items(), key=lambda kv: -kv[1]))
        print(f"           指令直方图: {hist}")


# --------------------------------------------------------------------------
# 示例程序
# --------------------------------------------------------------------------
SUM_PROGRAM = """
; ---- 求 1 + 2 + ... + 10，结果存入 DMEM[100] 并输出 ----
        LOADI R1, 0          ; R1 = sum = 0
        LOADI R2, 1          ; R2 = i = 1
        LOADI R3, 11         ; R3 = 上界(取不到)
loop:   SLT   R4, R2, R3     ; R4 = (i < 11)
        JZ    R4, done       ; 条件不成立就跳出循环
        ADD   R1, R1, R2     ; sum += i
        ADDI  R2, 1          ; i++
        JMP   loop
done:   STORE R1, 100        ; DMEM[100] = sum
        OUT   R1             ; 输出结果
        HALT
"""

# 数组求和：演示 LOAD/STORE 与内存访问模式（自修改地址由 ADDI 完成不了，
# 因此这里用展开的方式演示数据存储器读写，重点是 MEM 段的行为）
ARRAY_PROGRAM = """
; ---- 把 DMEM[0..4] 五个数累加，结果放 DMEM[10] ----
        LOAD  R1, 0
        LOAD  R2, 1
        ADD   R1, R1, R2
        LOAD  R2, 2
        ADD   R1, R1, R2
        LOAD  R2, 3
        ADD   R1, R1, R2
        LOAD  R2, 4
        ADD   R1, R1, R2
        STORE R1, 10
        OUT   R1
        HALT
"""

OVERFLOW_PROGRAM = """
; ---- 16 位补码溢出演示：32767 + 1 = -32768 ----
        LOADI R1, 255
        LOADI R2, 128
        ADD   R3, R1, R1     ; 510
loop:   ADD   R3, R3, R3     ; 反复翻倍，观察回绕
        ADDI  R2, -1
        JNZ   R2, loop
        OUT   R3
        HALT
"""


def demo_sum():
    print("=" * 66)
    print("演示 1：求 1+2+...+10（带逐周期 trace）")
    print("=" * 66)
    code = assemble(SUM_PROGRAM)
    print(f"机器码({len(code)} 条): " + " ".join(f"{w:04X}" for w in code))
    print("\n反汇编校验:")
    for i, w in enumerate(code):
        print(f"  [{i:>2}] {w:04X}  {disassemble(w)}")
    print("\n执行轨迹（只显示前 12 个周期）:")
    cpu = ToyCPU(code, trace=False)
    for _ in range(12):
        if cpu.halted:
            break
        cpu.trace = True
        cpu.step()
    cpu.trace = False
    cpu.run()
    print(f"  ... 共 {cpu.cycles} 个周期")
    print(f"\n输出端口: {cpu.output}")
    print(f"DMEM[100] = {cpu.dmem[100]}   （期望 55）")
    assert cpu.dmem[100] == 55 and cpu.output == [55], "求和结果错误"
    cpu.report()


def demo_array():
    print("\n" + "=" * 66)
    print("演示 2：数组求和，观察 LOAD/STORE 的 MEM 段行为")
    print("=" * 66)
    cpu = ToyCPU(assemble(ARRAY_PROGRAM))
    cpu.dmem[0:5] = [3, 14, 15, 92, 65]  # 预置数据
    cpu.run()
    print(f"输入数组: {[3, 14, 15, 92, 65]}  -> DMEM[10] = {cpu.dmem[10]} （期望 189）")
    assert cpu.dmem[10] == 189
    cpu.report()


def demo_overflow():
    print("\n" + "=" * 66)
    print("演示 3：16 位补码溢出（硬件不报错，只是悄悄回绕）")
    print("=" * 66)
    cpu = ToyCPU(assemble(OVERFLOW_PROGRAM))
    cpu.run()
    print(f"510 连续翻倍 128 次后 = {cpu.output[0]}")
    print("解释: 每次左移一位，有效位被挤出 16 位边界后丢失，最终归零/回绕。")
    print("这正是 C 语言 signed 溢出是未定义行为、而硬件只做截断的根源。")
    cpu.report()


def demo_cpi_comparison():
    """用同一段程序对比单周期 / 多周期 / 理想流水线的性能。"""
    print("\n" + "=" * 66)
    print("演示 4：单周期 vs 多周期 vs 流水线（同一程序的时间对比）")
    print("=" * 66)
    code = assemble(SUM_PROGRAM)
    cpu = ToyCPU(code)
    cpu.run()
    n = cpu.instr_count

    # 多周期：每类指令占用的周期数不同，时钟周期取最长单段
    mc_cycles_per_op = {"LOAD": 5, "STORE": 4, "HALT": 3, "JMP": 3, "JZ": 3,
                        "JNZ": 3, "OUT": 3, "LOADI": 4, "ADDI": 4,
                        "ADD": 4, "SUB": 4, "AND": 4, "OR": 4, "SLT": 4}
    mc_cycles = sum(mc_cycles_per_op[k] * v for k, v in cpu.op_stat.items())
    t_single = sum(ToyCPU.STAGE_DELAY.values())          # 8.0 ns
    t_multi = max(ToyCPU.STAGE_DELAY.values())           # 2.0 ns
    t_pipe = t_multi + 0.2                               # 流水线寄存器额外开销

    rows = [
        ("单周期", n, t_single, n * t_single),
        ("多周期", mc_cycles, t_multi, mc_cycles * t_multi),
        ("五级流水(理想)", n + 4, t_pipe, (n + 4) * t_pipe),
    ]
    print(f"{'实现方式':<16}{'周期数':>8}{'时钟周期(ns)':>14}{'总时间(ns)':>13}{'CPI':>8}")
    for name, cyc, tc, tot in rows:
        print(f"{name:<16}{cyc:>8}{tc:>14.1f}{tot:>13.1f}{cyc / n:>8.2f}")
    print("\n结论: 多周期靠缩短时钟周期取胜，流水线靠 CPI≈1 且时钟周期短双重取胜。")
    print("      注意 CPI 单独看没有意义，必须乘上时钟周期才是真实时间。")


if __name__ == "__main__":
    demo_sum()
    demo_array()
    demo_overflow()
    demo_cpi_comparison()
    print("\n全部演示通过，断言无报错。")
