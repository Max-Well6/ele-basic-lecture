# metastability.py —— 亚稳态 MTBF 计算 + 同步器蒙特卡洛验证（仅标准库）
# 对应讲义《数字电子技术》扩展二：异步电路与亚稳态
import math
import random


def mtbf(t_resolve, f_clk, f_data, tau=100e-12, t0=20e-12):
    """亚稳态平均无故障间隔（秒）。

    MTBF = exp(t_resolve / tau) / (T0 * f_clk * f_data)
      t_resolve  留给亚稳态自行衰减的时间（一个同步级 ≈ Tclk - tsu）
      tau        触发器的解析时间常数，由工艺决定（先进工艺约 10~30 ps）
      T0         孔径窗口系数
    指数项说明：多给一级同步器（t_resolve 翻倍），MTBF 呈指数级改善。
    """
    return math.exp(t_resolve / tau) / (t0 * f_clk * f_data)


def human(seconds):
    for unit, div in (("秒", 1), ("分", 60), ("时", 3600),
                      ("天", 86400), ("年", 365 * 86400)):
        if seconds < div * 1000 or unit == "年":
            return f"{seconds / div:.3g} {unit}"
    return f"{seconds:.3g} 秒"


def monte_carlo(cycles, f_clk, f_data, aperture, stages, tau, seed=0):
    """蒙特卡洛：异步数据跳变落进孔径窗口即产生亚稳态，
    再按 exp(-t/tau) 的概率判断它能否在 stages 级内衰减掉。"""
    rng = random.Random(seed)
    t_clk = 1.0 / f_clk
    p_toggle = min(1.0, f_data / f_clk)          # 每拍出现一次数据跳变的概率
    hits = escapes = 0
    for _ in range(cycles):
        if rng.random() >= p_toggle:
            continue
        phase = rng.random() * t_clk             # 跳变相对时钟沿的相位
        if phase >= aperture:                    # 落在安全区，正常采样
            continue
        hits += 1
        # 亚稳态残留到下一级输入端的概率随可用衰减时间指数下降
        if rng.random() < math.exp(-(stages * t_clk) / tau):
            escapes += 1
    return hits, escapes


if __name__ == "__main__":
    F_CLK, F_DATA = 1e9, 1e6                 # 1GHz 时钟，1MHz 异步事件
    TSU, TAU = 0.1e-9, 100e-12               # 慢速工艺角的 tau，最保守估计
    t_clk = 1 / F_CLK

    print(f"时钟 {F_CLK/1e9:.0f} GHz，异步事件 {F_DATA/1e6:.0f} MHz，"
          f"tau = {TAU*1e12:.0f} ps（慢角）")
    print("同步级数 | 可用衰减时间 | MTBF")
    for stages in (1, 2, 3):
        t_res = stages * t_clk - TSU
        print(f"   {stages}     | {t_res*1e9:>8.2f} ns  | "
              f"{human(mtbf(t_res, F_CLK, F_DATA, TAU)):>12}")
    print("1 级同步器不到 1 秒就出一次错，2 级撑几小时，3 级才够安全——")
    print("MTBF 对级数是指数敏感的，这就是打两拍(甚至三拍)的理由。\n")

    # 蒙特卡洛用刻意放大的玩具参数，否则真实概率太小、跑多少拍都撞不上
    print("蒙特卡洛（200 万拍，玩具参数：孔径=5%周期，tau=25%周期）：")
    print("同步级数 | 落入孔径次数 | 残留亚稳态(危险)次数")
    for stages in (1, 2, 3):
        hits, esc = monte_carlo(2_000_000, F_CLK, f_data=100e6,
                                aperture=0.05 * t_clk, stages=stages,
                                tau=0.25 * t_clk)
        print(f"   {stages}     | {hits:>10}   | {esc:>12}")
    print("落入孔径的次数与级数无关（取决于物理接口），但危险事件数随级数骤降。")
