# -*- coding: utf-8 -*-
"""脉动阵列（Systolic Array）矩阵乘法逐拍模拟 —— TPU 核心思想演示。

计算 C = A x B（均为 N x N），采用"输出驻留"（output-stationary）结构：
  - N x N 个 PE（处理单元）排成方阵，PE[i][j] 负责累加出 C[i][j]
  - A 的第 i 行从阵列左侧第 i 行注入，向右逐拍流动
  - B 的第 j 列从阵列顶部第 j 列注入，向下逐拍流动
  - 输入按"斜排"（skew）错开：第 i 行/列延迟 i 拍进入，
    保证 a[i][k] 与 b[k][j] 恰好在 PE[i][j] 相遇
  - 每拍每个 PE 做一次乘加：acc += a_in * b_in，然后把 a 右传、b 下传

数据像心脏泵血一样一拍一拍"脉动"穿过阵列，故名 systolic。
总拍数 = 3N - 2（N=3 时为 7 拍）。只用标准库。
"""


def systolic_matmul(A, B, verbose=True):
    n = len(A)
    acc = [[0] * n for _ in range(n)]       # 每个 PE 的累加器（驻留 C[i][j]）
    a_reg = [[None] * n for _ in range(n)]  # PE 中向右流动的 a 寄存器
    b_reg = [[None] * n for _ in range(n)]  # PE 中向下流动的 b 寄存器

    total_cycles = 3 * n - 2
    for t in range(total_cycles):
        # ---------- 1) 数据在阵列内向右/向下移动一格（从远端往近端搬）----------
        for i in range(n):
            for j in range(n - 1, 0, -1):
                a_reg[i][j] = a_reg[i][j - 1]      # a 右移
        for i in range(n - 1, 0, -1):
            for j in range(n):
                b_reg[i][j] = b_reg[i - 1][j]      # b 下移

        # ---------- 2) 边界注入：斜排输入 ----------
        # 第 i 行在第 t 拍注入 a[i][t-i]（若下标合法），否则注入空拍 None
        for i in range(n):
            k = t - i
            a_reg[i][0] = A[i][k] if 0 <= k < n else None
        # 第 j 列在第 t 拍注入 b[t-j][j]
        for j in range(n):
            k = t - j
            b_reg[0][j] = B[k][j] if 0 <= k < n else None

        # ---------- 3) 所有 PE 并行做乘加 ----------
        for i in range(n):
            for j in range(n):
                if a_reg[i][j] is not None and b_reg[i][j] is not None:
                    acc[i][j] += a_reg[i][j] * b_reg[i][j]

        # ---------- 4) 打印本拍状态 ----------
        if verbose:
            print("--- 第 {} 拍 ---".format(t + 1))
            for i in range(n):
                cells = []
                for j in range(n):
                    a = "-" if a_reg[i][j] is None else str(a_reg[i][j])
                    b = "-" if b_reg[i][j] is None else str(b_reg[i][j])
                    cells.append("[a={:>2} b={:>2}|acc={:>3}]".format(
                        a, b, acc[i][j]))
                print("  " + " ".join(cells))
    return acc


def plain_matmul(A, B):
    """朴素三重循环，用于对照验证。"""
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def main():
    A = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    B = [[1, 0, 2],
         [0, 1, 2],
         [1, 1, 1]]

    print("A x B, N = 3, 共 3N-2 = 7 拍")
    print("每个格子显示: 本拍流经的 a、b 和累加器 acc")
    print()
    C = systolic_matmul(A, B)

    expect = plain_matmul(A, B)
    print()
    print("脉动阵列结果:", C)
    print("朴素乘法结果:", expect)
    print("结果一致:", C == expect)
    print()
    print("要点: 每个输入数据只从内存读取一次, 就被 N 个 PE 复用 N 次,")
    print("N x N 阵列每拍完成 N^2 次乘加 —— 用数据流动换访存带宽。")


if __name__ == "__main__":
    main()
