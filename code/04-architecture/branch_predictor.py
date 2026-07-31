# -*- coding: utf-8 -*-
"""分支预测器对比实验：静态预测 / 1-bit / 2-bit 饱和计数器。

2-bit 饱和计数器状态机（0..3）：
  0 强不跳, 1 弱不跳, 2 弱跳, 3 强跳
  实际跳转则 +1（封顶 3），不跳则 -1（保底 0）；
  状态 >= 2 时预测"跳转"。
只用标准库。
"""


def predict_static(history, taken=True):
    """静态预测：永远猜 taken（或永远猜 not-taken）。"""
    correct = sum(1 for b in history if b == taken)
    return correct / len(history)


def predict_1bit(history, init=0):
    """1-bit 预测器：记住上一次的结果，猜下一次和上次一样。"""
    state = init  # 0 = 预测不跳, 1 = 预测跳
    correct = 0
    for actual in history:
        pred = (state == 1)
        if pred == actual:
            correct += 1
        state = 1 if actual else 0  # 直接改写为本次实际结果
    return correct / len(history)


def predict_2bit(history, init=1):
    """2-bit 饱和计数器：需要连续猜错两次才会翻转预测方向。"""
    state = init  # 0..3
    correct = 0
    for actual in history:
        pred = (state >= 2)
        if pred == actual:
            correct += 1
        if actual:
            state = min(3, state + 1)
        else:
            state = max(0, state - 1)
    return correct / len(history)


def make_loop_pattern(iters, trips):
    """模拟嵌套循环里的分支：每 trips 次跳转后跟 1 次不跳（循环退出）。"""
    pat = []
    for _ in range(iters):
        pat.extend([True] * trips + [False])
    return pat


def main():
    patterns = {
        "全 taken (TTTT...)": [True] * 100,
        "交替 (TNTN...)": [i % 2 == 0 for i in range(100)],
        "循环退出 (TTT N x25)": make_loop_pattern(25, 3),
        "长循环 (T x99, N x1)": make_loop_pattern(1, 99),
    }

    print("{:<22}{:>12}{:>12}{:>12}".format(
        "分支模式", "静态taken", "1-bit", "2-bit"))
    for name, hist in patterns.items():
        print("{:<22}{:>11.0%}{:>11.0%}{:>11.0%}".format(
            name,
            predict_static(hist),
            predict_1bit(hist),
            predict_2bit(hist)))

    print()
    print("观察：循环退出模式下 2-bit 明显优于 1-bit ——")
    print("1-bit 在每次循环退出时连错两次（退出 1 次 + 重入 1 次），")
    print("2-bit 的'惯性'让它在偶发的不跳后仍坚持预测跳转，只错一次。")


if __name__ == "__main__":
    main()
