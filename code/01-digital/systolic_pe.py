# systolic_pe.py —— 输出驻留型脉动阵列的逐拍 RTL 级模拟（仅标准库）
# 对应讲义《数字电子技术》扩展一：脉动阵列的数字实现
import random


def systolic_matmul(A, B, trace=False):
    """用 n×m 的 PE 网格计算 A(n×k) · B(k×m)，逐拍模拟寄存器行为。

    数据流：A 从左边界逐行注入、向右流动；B 从上边界逐列注入、向下流动。
    第 i 行 A 延迟 i 拍进入、第 j 列 B 延迟 j 拍进入（斜切 skew），
    这样 PE(i,j) 在第 t 拍恰好同时看到 A[i][t-i-j] 与 B[t-i-j][j]。
    """
    n, k, m = len(A), len(B), len(B[0])
    acc = [[0] * m for _ in range(n)]      # 每个 PE 的部分和寄存器
    a_reg = [[0] * m for _ in range(n)]    # PE 内横向流水寄存器
    b_reg = [[0] * m for _ in range(n)]    # PE 内纵向流水寄存器

    total_cycles = n + m + k - 1
    for t in range(total_cycles):
        # ---- 组合逻辑：算出本拍每个 PE 的输入 ----
        a_in = [[0] * m for _ in range(n)]
        b_in = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if j == 0:                       # 左边界：从外部注入
                    idx = t - i
                    a_in[i][j] = A[i][idx] if 0 <= idx < k else 0
                else:                            # 内部：来自左邻居的寄存器
                    a_in[i][j] = a_reg[i][j - 1]
                if i == 0:                       # 上边界：从外部注入
                    idx = t - j
                    b_in[i][j] = B[idx][j] if 0 <= idx < k else 0
                else:                            # 内部：来自上邻居的寄存器
                    b_in[i][j] = b_reg[i - 1][j]
        # ---- 时钟上升沿：所有 PE 同时做一次乘加并锁存流水值 ----
        for i in range(n):
            for j in range(m):
                acc[i][j] += a_in[i][j] * b_in[i][j]
        a_reg, b_reg = a_in, b_in
        if trace:
            busy = sum(1 for i in range(n) for j in range(m)
                       if a_in[i][j] and b_in[i][j])
            print(f"  t={t:>2}  活跃 PE 数={busy:>2}/{n * m}  "
                  f"PE(0,0)累加值={acc[0][0]}")
    return acc


def naive_matmul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    return [[sum(A[i][p] * B[p][j] for p in range(k)) for j in range(m)]
            for i in range(n)]


if __name__ == "__main__":
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = [[1, 0, 2], [0, 1, 3], [4, 5, 0]]
    print("3x3 脉动阵列逐拍执行（注意波前推进：活跃 PE 先增后减）")
    out = systolic_matmul(A, B, trace=True)
    print("脉动阵列结果:", out)
    print("朴素算法结果:", naive_matmul(A, B))
    assert out == naive_matmul(A, B)

    random.seed(0)
    for _ in range(200):                      # 随机对拍
        n, k, m = (random.randint(1, 5) for _ in range(3))
        A = [[random.randint(-9, 9) for _ in range(k)] for _ in range(n)]
        B = [[random.randint(-9, 9) for _ in range(m)] for _ in range(k)]
        assert systolic_matmul(A, B) == naive_matmul(A, B)
    print("PASS: 200 组随机矩阵对拍全部一致")

    # 利用率分析：阵列越大、K 越长，填充/排空开销占比越低
    print("\n阵列规模 | K   | 总拍数 | MAC 利用率")
    for size, kk in ((4, 4), (4, 64), (16, 16), (16, 256), (128, 256)):
        cycles = size + size + kk - 1
        util = (size * size * kk) / (size * size * cycles)
        print(f"{size:>3}x{size:<3}  | {kk:>3} | {cycles:>6} | {util:>8.1%}")
