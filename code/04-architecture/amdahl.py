# -*- coding: utf-8 -*-
"""Amdahl 定律与 Gustafson 定律计算表。

Amdahl:    Speedup = 1 / ((1-f) + f/s)
    f: 可并行（可加速）部分占比, s: 该部分的加速倍数
Gustafson: Speedup = (1-f) + f*s
    以"放大问题规模"的视角看并行收益
只用标准库。
"""


def amdahl(f, s):
    """Amdahl 定律：固定问题规模下的整体加速比。"""
    return 1.0 / ((1.0 - f) + f / s)


def gustafson(f, s):
    """Gustafson 定律：问题规模随处理器数扩大时的加速比。"""
    return (1.0 - f) + f * s


def main():
    fractions = [0.5, 0.9, 0.95, 0.99]   # 可并行部分占比
    speeds = [2, 4, 8, 16, 64, 1024]     # 并行部分加速倍数（处理器数）

    print("=== Amdahl 定律：Speedup = 1 / ((1-f) + f/s) ===")
    header = "f\\s   " + "".join("{:>9}".format(s) for s in speeds)
    print(header)
    for f in fractions:
        row = "{:<6}".format(f)
        row += "".join("{:>9.2f}".format(amdahl(f, s)) for s in speeds)
        print(row)

    print()
    print("=== Gustafson 定律：Speedup = (1-f) + f*s ===")
    print(header)
    for f in fractions:
        row = "{:<6}".format(f)
        row += "".join("{:>9.2f}".format(gustafson(f, s)) for s in speeds)
        print(row)

    # 结论演示：串行部分决定加速上限
    print()
    print("=== Amdahl 加速上限（s -> 无穷）: 1/(1-f) ===")
    for f in fractions:
        print("f = {:<5} 上限 = {:.1f}x".format(f, 1.0 / (1.0 - f)))


if __name__ == "__main__":
    main()
