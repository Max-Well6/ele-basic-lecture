"""
二极管伏安特性与非线性电路工作点求解
==========================================

本程序演示模拟电子技术中最基础也最重要的一个思想：
    "非线性器件 + 线性网络" 的工作点，必须用数值方法求解。

包含四个部分：
    1. 肖克利方程正向 I-V 特性表
    2. 牛顿迭代法求解 "电压源 + 电阻 + 二极管" 串联电路的静态工作点 Q
    3. 与工程近似模型（恒压降 0.7V）的误差对比
    4. 半波整流 + 电容滤波的时域迭代仿真（纹波估算）

只使用 Python 标准库（math），可直接运行：
    python diode_iv.py
"""

import math

# ---------------------------------------------------------------
# 物理常数与器件参数
# ---------------------------------------------------------------
K_BOLTZMANN = 1.380649e-23   # 玻尔兹曼常数 J/K
Q_ELECTRON = 1.602176634e-19  # 电子电荷 C


def thermal_voltage(temp_c=27.0):
    """热电压 VT = kT/q，室温 27C 约等于 25.9 mV。"""
    temp_k = temp_c + 273.15
    return K_BOLTZMANN * temp_k / Q_ELECTRON


class Diode:
    """
    肖克利二极管模型:  I = Is * (exp(V / (n*VT)) - 1)

    Is : 反向饱和电流，小信号硅管典型 1e-12 ~ 1e-14 A
    n  : 发射系数（理想因子），硅管取 1 ~ 2
    """

    def __init__(self, i_s=1e-12, n=1.0, temp_c=27.0, name="1N4148"):
        self.i_s = i_s
        self.n = n
        self.temp_c = temp_c
        self.name = name
        self.vt = thermal_voltage(temp_c)

    def current(self, v):
        """给定端电压返回电流(A)。对指数做限幅，避免 overflow。"""
        x = v / (self.n * self.vt)
        if x > 200.0:          # exp(200) 已超出双精度可用范围
            x = 200.0
        return self.i_s * (math.exp(x) - 1.0)

    def conductance(self, v):
        """微分电导 g = dI/dV = I_s/(n*VT) * exp(V/(n*VT))，即小信号电阻的倒数。"""
        x = v / (self.n * self.vt)
        if x > 200.0:
            x = 200.0
        return self.i_s / (self.n * self.vt) * math.exp(x)

    def dynamic_resistance(self, v):
        """小信号动态电阻 rd = n*VT / I，近似 26mV/I(mA) 欧姆。"""
        g = self.conductance(v)
        return 1.0 / g if g > 0 else float("inf")


# ---------------------------------------------------------------
# 1. 正向伏安特性表
# ---------------------------------------------------------------
def print_iv_table(diode):
    print("=" * 66)
    print(f"【1】二极管 {diode.name} 正向伏安特性  (Is={diode.i_s:.1e} A, "
          f"n={diode.n}, T={diode.temp_c}C, VT={diode.vt * 1000:.2f} mV)")
    print("=" * 66)
    print(f"{'V (V)':>8} | {'I (A)':>13} | {'I (mA)':>11} | {'rd (ohm)':>11}")
    print("-" * 66)
    for i in range(13):
        v = 0.05 * i          # 0 ~ 0.60 V
        current = diode.current(v)
        rd = diode.dynamic_resistance(v)
        rd_str = f"{rd:11.2f}" if rd < 1e9 else "        inf"
        print(f"{v:8.3f} | {current:13.4e} | {current * 1e3:11.5f} | {rd_str}")
    print("-" * 66)
    print("结论：电压每增加约 60 mV，电流增大 10 倍（十倍频程 = 2.3*n*VT）。")
    print()


# ---------------------------------------------------------------
# 2. 牛顿迭代求工作点
# ---------------------------------------------------------------
def solve_series_diode(vs, r, diode, v0=0.6, tol=1e-12, max_iter=100, verbose=True):
    """
    求解电路:  Vs --- R --- (+)二极管(-) --- GND

    KCL 方程:   f(V) = (Vs - V)/R - I_d(V) = 0
    牛顿迭代:   V_{k+1} = V_k - f(V_k) / f'(V_k)
                f'(V) = -1/R - g_d(V)

    返回 (Vd, Id, 迭代次数)
    """
    v = v0
    if verbose:
        print(f"{'iter':>4} | {'Vd (V)':>10} | {'Id (mA)':>12} | {'f(V) 残差':>13}")
        print("-" * 52)
    for k in range(1, max_iter + 1):
        f = (vs - v) / r - diode.current(v)
        fp = -1.0 / r - diode.conductance(v)
        dv = -f / fp
        # 阻尼：限制单步电压变化，保证指数函数下的收敛稳定性
        max_step = 0.1
        if abs(dv) > max_step:
            dv = math.copysign(max_step, dv)
        v += dv
        if verbose:
            print(f"{k:4d} | {v:10.6f} | {diode.current(v) * 1e3:12.6f} | {abs(f):13.3e}")
        if abs(dv) < tol or abs(f) < tol:
            return v, diode.current(v), k
    return v, diode.current(v), max_iter


def demo_operating_point():
    print("=" * 66)
    print("【2】牛顿迭代求静态工作点  (Vs = 5 V, R = 1 kOhm)")
    print("=" * 66)
    d = Diode()
    vs, r = 5.0, 1000.0
    vd, idc, n_iter = solve_series_diode(vs, r, d)
    print("-" * 52)
    print(f"精确解 : Vd = {vd:.6f} V,  Id = {idc * 1e3:.6f} mA,  迭代 {n_iter} 次")

    # 工程近似：恒压降模型
    vd_approx = 0.7
    id_approx = (vs - vd_approx) / r
    print(f"近似解 : Vd = {vd_approx:.6f} V,  Id = {id_approx * 1e3:.6f} mA  (恒压降 0.7V 模型)")
    err = abs(id_approx - idc) / idc * 100
    print(f"电流误差: {err:.2f}%  ->  Vs 远大于 0.7V 时近似模型完全够用")
    print()
    return d


# ---------------------------------------------------------------
# 3. 不同电源电压下的工作点扫描
# ---------------------------------------------------------------
def sweep_supply(diode):
    print("=" * 66)
    print("【3】电源电压扫描 (R = 1 kOhm)：观察 Vd 的\"钳位\"特性")
    print("=" * 66)
    print(f"{'Vs (V)':>8} | {'Vd (V)':>9} | {'Id (mA)':>10} | {'恒压降Id':>10} | {'误差%':>8}")
    print("-" * 66)
    r = 1000.0
    for vs in [0.5, 1.0, 2.0, 3.0, 5.0, 9.0, 12.0, 15.0]:
        vd, idc, _ = solve_series_diode(vs, r, diode, verbose=False)
        id_ap = max((vs - 0.7) / r, 0.0)
        err = abs(id_ap - idc) / idc * 100 if idc > 1e-15 else 0.0
        print(f"{vs:8.2f} | {vd:9.5f} | {idc * 1e3:10.5f} | {id_ap * 1e3:10.5f} | {err:8.2f}")
    print("-" * 66)
    print("结论：Vs 越大，Vd 变化越小（对数压缩），这正是二极管能做钳位/限幅的原因。")
    print()


# ---------------------------------------------------------------
# 4. 半波整流 + 电容滤波时域仿真
# ---------------------------------------------------------------
def rectifier_sim(vpeak=12.0, freq=50.0, c=470e-6, r_load=1000.0,
                  diode=None, n_cycle=4, steps_per_cycle=2000):
    """
    后向欧拉法迭代求解:
        C * dVo/dt = I_diode(Vin - Vo) - Vo / R_load

    为避免解隐式方程，此处使用足够小的步长做显式积分（步长 = T/2000）。
    """
    if diode is None:
        diode = Diode(i_s=1e-9, n=1.6, name="1N4007")  # 整流管 Is 更大
    t_end = n_cycle / freq
    dt = 1.0 / (freq * steps_per_cycle)
    vo = 0.0
    t = 0.0
    vo_min_last, vo_max_last = 1e9, -1e9
    samples = []
    while t < t_end:
        vin = vpeak * math.sin(2 * math.pi * freq * t)
        i_d = diode.current(vin - vo)
        # 反偏时电流极小，直接置零可加速收敛
        if vin - vo < 0:
            i_d = 0.0
        dvo = (i_d - vo / r_load) / c * dt
        vo += dvo
        t += dt
        if t > t_end - 1.0 / freq:          # 只统计最后一个周期
            vo_min_last = min(vo_min_last, vo)
            vo_max_last = max(vo_max_last, vo)
            samples.append((t, vin, vo))
    ripple = vo_max_last - vo_min_last
    vo_avg = sum(s[2] for s in samples) / len(samples)

    print("=" * 66)
    print("【4】半波整流 + 电容滤波 时域仿真")
    print("=" * 66)
    print(f"输入: {vpeak} V 峰值 / {freq} Hz   滤波电容: {c * 1e6:.0f} uF   负载: {r_load:.0f} Ohm")
    print(f"输出平均值 Vo(avg) = {vo_avg:.3f} V")
    print(f"输出峰值   Vo(max) = {vo_max_last:.3f} V")
    print(f"输出谷值   Vo(min) = {vo_min_last:.3f} V")
    print(f"纹波峰峰值 Vpp     = {ripple:.3f} V")
    # 工程估算公式 Vpp ~ Io/(f*C)
    io = vo_avg / r_load
    ripple_est = io / (freq * c)
    print(f"工程估算   Vpp ~ Io/(f*C) = {ripple_est:.3f} V   (误差 "
          f"{abs(ripple_est - ripple) / ripple * 100:.1f}%)")
    print()
    print(f"{'相位(deg)':>10} | {'Vin (V)':>9} | {'Vo (V)':>9}")
    print("-" * 36)
    step = max(1, len(samples) // 12)
    for i in range(0, len(samples), step):
        t_s, vin, vo_s = samples[i]
        phase = (t_s * freq % 1.0) * 360
        print(f"{phase:10.1f} | {vin:9.3f} | {vo_s:9.3f}")
    print()


def main():
    d = Diode()
    print_iv_table(d)
    d = demo_operating_point()
    sweep_supply(d)
    rectifier_sim()
    print("=" * 66)
    print("要点回顾：")
    print("  1) 非线性器件电路 -> 牛顿迭代，核心是 f(V)=0 与雅可比 f'(V)")
    print("  2) 小信号电阻 rd = VT/I，是把非线性电路\"线性化\"的桥梁")
    print("  3) 工程近似模型（0.7V 恒压降）在大信号下误差可接受")
    print("  4) 电容滤波纹波 Vpp ~ Io/(f*C)，C 越大纹波越小")
    print("=" * 66)


if __name__ == "__main__":
    main()
