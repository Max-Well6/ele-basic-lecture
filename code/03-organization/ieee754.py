"""
IEEE 754 单精度/双精度浮点数手工编解码器
=========================================
只用标准库。核心目的：把「浮点数为什么不精确」从玄学变成可以逐位验算的算术。

IEEE 754 binary32 位布局（共 32 位）：
    [31]      符号位 S      1 位
    [30:23]   阶码 E        8 位（移码，偏置 127）
    [22:0]    尾数 M       23 位（规格化时隐含前导 1）

真值公式（规格化数，1 <= E <= 254）：
    value = (-1)^S * 1.M * 2^(E - 127)
非规格化数（E == 0）：
    value = (-1)^S * 0.M * 2^(-126)

运行：python ieee754.py
"""

import struct


# --------------------------------------------------------------------------
# 格式参数：把单/双精度统一成一张参数表，避免写两套代码
# --------------------------------------------------------------------------
FORMATS = {
    32: {"exp_bits": 8, "frac_bits": 23, "bias": 127, "pack": ">f"},
    64: {"exp_bits": 11, "frac_bits": 52, "bias": 1023, "pack": ">d"},
}


def encode(value: float, width: int = 32) -> int:
    """把 Python float 编码成 IEEE 754 位串（返回整数形式的位模式）。

    这里刻意不调用 struct，而是纯算术实现，便于看清每一步。
    """
    fmt = FORMATS[width]
    exp_bits, frac_bits, bias = fmt["exp_bits"], fmt["frac_bits"], fmt["bias"]
    exp_max = (1 << exp_bits) - 1

    # 1) 处理符号位：先取出符号，后面只处理绝对值
    sign = 0
    if value < 0 or (value == 0.0 and str(value)[0] == "-"):
        sign = 1
        value = -value

    # 2) 特殊值：0 / inf / NaN
    if value == 0.0:
        return sign << (exp_bits + frac_bits)
    if value != value:  # NaN 的经典判断：自己不等于自己
        return (sign << (exp_bits + frac_bits)) | (exp_max << frac_bits) | (1 << (frac_bits - 1))
    if value == float("inf"):
        return (sign << (exp_bits + frac_bits)) | (exp_max << frac_bits)

    # 3) 规格化：不断乘/除 2，把尾数拉到 [1, 2) 区间
    e = 0
    while value >= 2.0:
        value /= 2.0
        e += 1
    while value < 1.0:
        value *= 2.0
        e -= 1

    E = e + bias
    if E >= exp_max:  # 上溢 -> 无穷大
        return (sign << (exp_bits + frac_bits)) | (exp_max << frac_bits)

    if E <= 0:
        # 非规格化数：阶码字段为 0，尾数不再有隐含 1
        frac = int(round(value * (1 << frac_bits) / (1 << (1 - E))))
        return (sign << (exp_bits + frac_bits)) | frac

    # 4) 取尾数小数部分（去掉隐含的整数位 1），四舍五入到 frac_bits 位
    frac = int(round((value - 1.0) * (1 << frac_bits)))
    if frac == (1 << frac_bits):  # 进位导致尾数溢出，阶码 +1
        frac = 0
        E += 1
    return (sign << (exp_bits + frac_bits)) | (E << frac_bits) | frac


def decode(bits: int, width: int = 32) -> float:
    """把位模式还原成实数值（纯算术，不用 struct）。"""
    fmt = FORMATS[width]
    exp_bits, frac_bits, bias = fmt["exp_bits"], fmt["frac_bits"], fmt["bias"]
    exp_max = (1 << exp_bits) - 1

    sign = (bits >> (exp_bits + frac_bits)) & 1
    E = (bits >> frac_bits) & exp_max
    frac = bits & ((1 << frac_bits) - 1)

    if E == exp_max:
        return float("nan") if frac else (float("-inf") if sign else float("inf"))
    if E == 0:
        # 非规格化数：0.M * 2^(1-bias)
        val = (frac / (1 << frac_bits)) * (2.0 ** (1 - bias))
    else:
        val = (1.0 + frac / (1 << frac_bits)) * (2.0 ** (E - bias))
    return -val if sign else val


def explain(value: float, width: int = 32) -> str:
    """输出一行人类可读的位域拆解，用于课堂演示。"""
    fmt = FORMATS[width]
    exp_bits, frac_bits, bias = fmt["exp_bits"], fmt["frac_bits"], fmt["bias"]
    bits = encode(value, width)
    s = (bits >> (exp_bits + frac_bits)) & 1
    e = (bits >> frac_bits) & ((1 << exp_bits) - 1)
    f = bits & ((1 << frac_bits) - 1)
    return (
        f"{value!r:>24} -> 0x{bits:0{width // 4}X}  "
        f"S={s} E={e}(实际指数 {e - bias if e else '非规格化'}) "
        f"M=0b{f:0{frac_bits}b}"
    )


def struct_bits(value: float, width: int = 32) -> int:
    """用 struct 得到标准答案，作为交叉验证的基准。"""
    packed = struct.pack(FORMATS[width]["pack"], value)
    return int.from_bytes(packed, "big")


def selftest() -> None:
    """与 struct 模块逐位对照，确认手工实现正确。"""
    cases = [0.0, 1.0, -1.0, 0.5, 2.0, 3.14159, -273.15, 1e-3, 6.02e23, 1.0 / 3.0]
    print("=== 手工编码 vs struct 标准实现（binary32）===")
    for v in cases:
        mine = encode(v, 32)
        ref = struct_bits(v, 32)
        ok = "OK " if mine == ref else "FAIL"
        print(f"[{ok}] {v!r:>22}  mine=0x{mine:08X}  struct=0x{ref:08X}")
        assert mine == ref, f"编码不一致: {v}"

    print("\n=== 位域拆解 ===")
    for v in [1.0, 0.1, -2.5, 1.5e-45]:
        print(explain(v))

    print("\n=== 解码回环（binary64 精度足以完全还原）===")
    for v in cases:
        back = decode(encode(v, 64), 64)
        print(f"{v!r:>22} -> {back!r}")
        assert back == v or (v != v), f"解码不一致: {v}"


def why_01_plus_02() -> None:
    """经典问题：为什么 0.1 + 0.2 != 0.3。"""
    print("\n=== 0.1 + 0.2 != 0.3 的位级真相 ===")
    for v in (0.1, 0.2, 0.3, 0.1 + 0.2):
        print(f"{v!r:>22}  bits=0x{struct_bits(v, 64):016X}")
    print(f"0.1 + 0.2 == 0.3 ? {0.1 + 0.2 == 0.3}")
    print(f"实际差值: {0.1 + 0.2 - 0.3!r}")
    # 0.1 在二进制下是无限循环小数 0.0001100110011...，
    # 截断到 52 位尾数必然引入误差，两次舍入误差叠加后与 0.3 的舍入结果不同。
    print("结论: 十进制有限小数在二进制下可能无限循环，舍入误差不可避免。")
    print("工程做法: 浮点比较用 abs(a-b) < eps；金额计算用整数分或 decimal。")


def machine_epsilon() -> None:
    """机器精度：1.0 之后的下一个可表示数与 1.0 的距离。"""
    print("\n=== 机器精度 ===")
    eps32 = decode(encode(1.0, 32) + 1, 32) - 1.0
    eps64 = decode(encode(1.0, 64) + 1, 64) - 1.0
    print(f"binary32 eps = {eps32!r}  (2^-23 = {2 ** -23!r})")
    print(f"binary64 eps = {eps64!r}  (2^-52 = {2 ** -52!r})")
    # AI 训练里的 fp16/bf16 就是在拿精度换带宽：位宽越小，eps 越大。
    print("推论: fp16 尾数仅 10 位，eps≈9.8e-4，所以混合精度训练要用 fp32 存主权重。")


if __name__ == "__main__":
    selftest()
    why_01_plus_02()
    machine_epsilon()
    print("\n全部断言通过。")
