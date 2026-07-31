"""
频率响应分析：RC 滤波器、波特图与有源滤波器
================================================

模拟电路的"第二个维度"：增益不是常数，而是频率的函数 A(jw)。
本程序用 cmath 做复数运算，输出文本形式的波特图数据。

包含：
    1. 一阶 RC 低通/高通的幅频、相频特性（含 -3dB 点、-20dB/十倍频斜率验证）
    2. ASCII 波特图绘制（不依赖 matplotlib）
    3. 放大电路的完整频响：下限频率 fL（耦合电容）+ 上限频率 fH（结电容/米勒效应）
    4. 二阶 Sallen-Key 有源低通滤波器（Butterworth 设计）
    5. 增益带宽积 GBW 与运放闭环带宽的关系

仅使用标准库 cmath / math，直接运行：
    python rc_filter.py
"""

import cmath
import math

TWO_PI = 2 * math.pi


# =================================================================
# 通用工具
# =================================================================
def db(x):
    """线性幅值转分贝。"""
    return 20 * math.log10(abs(x)) if abs(x) > 1e-30 else -300.0


def deg(z):
    """复数相角，单位度。"""
    return math.degrees(cmath.phase(z))


def log_sweep(f_start, f_stop, points_per_decade=4):
    """对数频率扫描，生成频点列表。"""
    n_dec = math.log10(f_stop / f_start)
    n = int(round(n_dec * points_per_decade))
    return [f_start * (10 ** (i / points_per_decade)) for i in range(n + 1)]


def ascii_bode(freqs, gains_db, width=52, title="幅频特性"):
    """用字符画出波特图，横轴为对数频率，纵轴为 dB。"""
    gmin, gmax = min(gains_db), max(gains_db)
    if gmax - gmin < 1e-6:
        gmax = gmin + 1.0
    # 上下各留一点余量并取整
    gmax = math.ceil(gmax / 10) * 10
    gmin = math.floor(gmin / 10) * 10
    print(f"  {title}   (纵轴 {gmin:.0f} ~ {gmax:.0f} dB)")
    print("  " + "-" * (width + 14))
    for f, g in zip(freqs, gains_db):
        pos = int((g - gmin) / (gmax - gmin) * (width - 1))
        pos = max(0, min(width - 1, pos))
        bar = " " * pos + "*"
        print(f"  {f:9.1f} Hz |{bar:<{width}}| {g:7.2f}")
    print("  " + "-" * (width + 14))


# =================================================================
# 1. 一阶 RC 滤波器
# =================================================================
class RCFilter:
    """
    一阶 RC 滤波器。

    低通:  H(jw) = 1 / (1 + jw*R*C)      转折频率 fc = 1/(2*pi*R*C)
    高通:  H(jw) = jw*R*C / (1 + jw*R*C)
    """

    def __init__(self, r=1e3, c=159e-9, kind="lowpass"):
        self.r = r
        self.c = c
        self.kind = kind
        self.fc = 1.0 / (TWO_PI * r * c)

    def transfer(self, f):
        """返回频率 f (Hz) 处的复数传递函数。"""
        s = 1j * TWO_PI * f
        tau = self.r * self.c
        if self.kind == "lowpass":
            return 1.0 / (1.0 + s * tau)
        else:                                  # highpass
            return (s * tau) / (1.0 + s * tau)

    def table(self, freqs):
        rows = []
        for f in freqs:
            h = self.transfer(f)
            rows.append((f, abs(h), db(h), deg(h)))
        return rows


def demo_rc_filter():
    print("=" * 72)
    print("【1】一阶 RC 低通滤波器频率响应")
    print("=" * 72)
    lpf = RCFilter(r=1e3, c=159e-9, kind="lowpass")
    print(f"R = {lpf.r:.0f} Ohm, C = {lpf.c*1e9:.0f} nF  ->  "
          f"fc = 1/(2*pi*R*C) = {lpf.fc:.2f} Hz")
    print("-" * 72)
    print(f"{'f (Hz)':>11} | {'f/fc':>8} | {'|H|':>8} | {'|H| dB':>9} | {'相位(deg)':>10}")
    print("-" * 72)
    for f in [10, 50, 100, 500, lpf.fc, 2000, 5000, 10000, 50000, 100000]:
        h = lpf.transfer(f)
        print(f"{f:11.2f} | {f/lpf.fc:8.3f} | {abs(h):8.5f} | {db(h):9.3f} | {deg(h):10.2f}")
    print("-" * 72)
    h_fc = lpf.transfer(lpf.fc)
    print(f"验证 fc 处: |H| = {abs(h_fc):.5f} (理论 1/sqrt(2) = {1/math.sqrt(2):.5f}), "
          f"{db(h_fc):.3f} dB, 相位 {deg(h_fc):.2f} 度")
    # 验证 -20dB/decade 斜率
    g1, g2 = db(lpf.transfer(10 * lpf.fc)), db(lpf.transfer(100 * lpf.fc))
    print(f"验证滚降斜率: 从 10fc 到 100fc 衰减 {g2-g1:.2f} dB/十倍频 (理论 -20 dB)")
    print()

    freqs = log_sweep(10, 1e5, points_per_decade=3)
    gains = [db(lpf.transfer(f)) for f in freqs]
    ascii_bode(freqs, gains, title="RC 低通 ASCII 波特图")
    print()

    print("=" * 72)
    print("     一阶 RC 高通滤波器")
    print("=" * 72)
    hpf = RCFilter(r=1e3, c=159e-9, kind="highpass")
    print(f"{'f (Hz)':>11} | {'|H| dB':>9} | {'相位(deg)':>10}   (fc = {hpf.fc:.1f} Hz)")
    print("-" * 72)
    for f in [10, 100, 500, hpf.fc, 5000, 20000, 100000]:
        h = hpf.transfer(f)
        print(f"{f:11.2f} | {db(h):9.3f} | {deg(h):10.2f}")
    print("-" * 72)
    print("高通在低频段以 +20dB/十倍频上升，相位从 +90 度趋向 0 度。")
    print()


# =================================================================
# 2. 放大电路完整频响（下限 + 上限频率）
# =================================================================
class AmplifierResponse:
    """
    单级共射放大器的简化频率响应模型：

        A(jw) = Am * [ jw*tau_L / (1 + jw*tau_L) ] * [ 1 / (1 + jw*tau_H) ]
                        低频（耦合电容）              高频（结电容/米勒电容）

    fL = 1/(2*pi*tau_L)   下限截止频率
    fH = 1/(2*pi*tau_H)   上限截止频率
    BW = fH - fL          通频带
    """

    def __init__(self, am=-150.0, fl=20.0, fh=200e3):
        self.am = am
        self.fl = fl
        self.fh = fh

    def transfer(self, f):
        if f <= 0:
            return 0j
        low = (1j * f) / (1j * f + self.fl)      # 低频段：耦合电容形成高通
        high = 1.0 / (1.0 + 1j * f / self.fh)    # 高频段：结电容形成低通
        return self.am * low * high

    def bandwidth(self):
        return self.fh - self.fl


def miller_capacitance(cbc, av):
    """
    米勒定理：跨接在输入输出之间的电容 Cbc，
    等效到输入端为 Cbc*(1+|Av|)，这是限制共射级带宽的元凶。
    """
    return cbc * (1 + abs(av))


def demo_amplifier_response():
    print("=" * 72)
    print("【2】放大电路完整频率响应与米勒效应")
    print("=" * 72)

    # 先算米勒电容导致的 fH
    rs_eff = 1000.0        # 信号源等效内阻（并联后）
    cbe, cbc = 20e-12, 4e-12
    av = -150.0
    cm = miller_capacitance(cbc, av)
    c_in_total = cbe + cm
    fh = 1.0 / (TWO_PI * rs_eff * c_in_total)
    print(f"结电容: Cbe = {cbe*1e12:.0f} pF, Cbc = {cbc*1e12:.0f} pF, 中频增益 Av = {av:.0f}")
    print(f"米勒等效输入电容 CM = Cbc*(1+|Av|) = {cm*1e12:.1f} pF")
    print(f"总输入电容 Ci = Cbe + CM = {c_in_total*1e12:.1f} pF  "
          f"(米勒电容占 {cm/c_in_total*100:.1f}%!)")
    print(f"上限频率 fH = 1/(2*pi*Rs*Ci) = {fh/1e3:.2f} kHz")
    print()
    print("--- 增益越大，带宽越窄：增益带宽积近似为常数 ---")
    print(f"{'|Av|':>8} | {'CM (pF)':>9} | {'Ci (pF)':>9} | {'fH (kHz)':>10} | {'GBW (MHz)':>11}")
    print("-" * 72)
    for a in [10, 25, 50, 100, 150, 300]:
        cm_i = miller_capacitance(cbc, a)
        ci = cbe + cm_i
        fh_i = 1.0 / (TWO_PI * rs_eff * ci)
        print(f"{a:8d} | {cm_i*1e12:9.1f} | {ci*1e12:9.1f} | {fh_i/1e3:10.2f} | "
              f"{a*fh_i/1e6:11.3f}")
    print("-" * 72)
    print("结论：这就是共基组态（无米勒效应）和 Cascode 结构存在的理由。")
    print()

    amp = AmplifierResponse(am=-150.0, fl=20.0, fh=fh)
    print(f"完整频响: |Am| = {abs(amp.am):.0f} ({db(amp.am):.2f} dB), "
          f"fL = {amp.fl:.1f} Hz, fH = {amp.fh/1e3:.2f} kHz, "
          f"BW = {amp.bandwidth()/1e3:.2f} kHz")
    print("-" * 72)
    print(f"{'f (Hz)':>12} | {'|A| dB':>9} | {'相对中频 dB':>12} | {'相位(deg)':>10}")
    print("-" * 72)
    mid_db = db(amp.am)
    for f in [1, 5, amp.fl, 100, 1e3, 1e4, 1e5, amp.fh, 1e6, 1e7]:
        h = amp.transfer(f)
        print(f"{f:12.1f} | {db(h):9.3f} | {db(h)-mid_db:12.3f} | {deg(h):10.2f}")
    print("-" * 72)
    print("在 fL 和 fH 处，增益均下降 3 dB（约 0.707 倍），这就是通频带的定义。")
    print()

    freqs = log_sweep(1, 1e7, points_per_decade=2)
    gains = [db(amp.transfer(f)) for f in freqs]
    ascii_bode(freqs, gains, title="放大器全频段 ASCII 波特图")
    print()


# =================================================================
# 3. 二阶 Sallen-Key 有源低通滤波器
# =================================================================
class SallenKeyLPF:
    """
    二阶 Sallen-Key 低通（单位增益结构）：

        H(s) = wn^2 / (s^2 + (wn/Q)*s + wn^2)

        wn = 1/sqrt(R1*R2*C1*C2)
        Q  = sqrt(R1*R2*C1*C2) / (C1*(R1+R2))    [单位增益时]

    Butterworth 最平坦响应: Q = 0.7071
    Chebyshev / Bessel 只是 Q 取值不同
    """

    def __init__(self, r1=10e3, r2=10e3, c1=22.5e-9, c2=11.2e-9):
        self.r1, self.r2, self.c1, self.c2 = r1, r2, c1, c2
        self.wn = 1.0 / math.sqrt(r1 * r2 * c1 * c2)
        self.fn = self.wn / TWO_PI
        self.q = math.sqrt(r1 * r2 * c1 * c2) / (c2 * (r1 + r2))

    def transfer(self, f):
        s = 1j * TWO_PI * f
        wn, q = self.wn, self.q
        return wn * wn / (s * s + (wn / q) * s + wn * wn)

    @staticmethod
    def design_butterworth(fc, c=10e-9):
        """
        给定截止频率设计 Butterworth（Q=0.7071）Sallen-Key 低通。
        取 R1=R2=R，则 Q = sqrt(C1*C2)/(C2*2) ... 这里用 C1 = 2*C2 的经典取法。
        R = 1/(2*pi*fc*sqrt(C1*C2))
        """
        c2 = c
        c1 = 2 * c              # C1 = 2*C2 => Q = 0.7071
        r = 1.0 / (TWO_PI * fc * math.sqrt(c1 * c2))
        return SallenKeyLPF(r1=r, r2=r, c1=c1, c2=c2)


def demo_active_filter():
    print("=" * 72)
    print("【3】二阶 Sallen-Key 有源低通滤波器（Butterworth 设计）")
    print("=" * 72)
    target_fc = 1000.0
    f = SallenKeyLPF.design_butterworth(target_fc, c=10e-9)
    print(f"设计目标: fc = {target_fc:.0f} Hz")
    print(f"元件取值: R1 = R2 = {f.r1:.1f} Ohm ({f.r1/1e3:.3f} kOhm), "
          f"C1 = {f.c1*1e9:.1f} nF, C2 = {f.c2*1e9:.1f} nF")
    print(f"实际特性: fn = {f.fn:.2f} Hz,  Q = {f.q:.4f} "
          f"(Butterworth 目标 {1/math.sqrt(2):.4f})")
    print("-" * 72)
    print(f"{'f (Hz)':>10} | {'|H| dB':>9} | {'相位(deg)':>10} | {'一阶对比 dB':>13}")
    print("-" * 72)
    first_order = RCFilter(r=1e3, c=1.0 / (TWO_PI * 1e3 * 1e3), kind="lowpass")
    for freq in [10, 100, 500, 1000, 2000, 5000, 10000, 50000]:
        h = f.transfer(freq)
        h1 = first_order.transfer(freq)
        print(f"{freq:10.1f} | {db(h):9.3f} | {deg(h):10.2f} | {db(h1):13.3f}")
    print("-" * 72)
    g1, g2 = db(f.transfer(10e3)), db(f.transfer(100e3))
    print(f"二阶滚降斜率: {g2-g1:.2f} dB/十倍频 (理论 -40 dB) —— 比一阶陡一倍")
    print(f"fc 处衰减: {db(f.transfer(f.fn)):.3f} dB (Butterworth 恰为 -3 dB)")
    print()

    print("--- 不同 Q 值的影响（同一 fn = 1 kHz）---")
    print(f"{'Q':>8} | {'类型':<14} | {'fn 处 dB':>10} | {'峰值 dB':>9} | {'峰值频率 Hz':>12}")
    print("-" * 72)
    for q, name in [(0.5, "临界阻尼"), (0.577, "Bessel"), (0.7071, "Butterworth"),
                    (1.0, "轻微峰化"), (2.0, "Chebyshev"), (5.0, "强谐振")]:
        filt = SallenKeyLPF()
        filt.wn = TWO_PI * 1000
        filt.fn = 1000
        filt.q = q
        best_f, best_g = 0, -1e9
        for freq in log_sweep(10, 10000, points_per_decade=60):
            g = db(filt.transfer(freq))
            if g > best_g:
                best_g, best_f = g, freq
        print(f"{q:8.4f} | {name:<14} | {db(filt.transfer(1000)):10.3f} | "
              f"{best_g:9.3f} | {best_f:12.1f}")
    print("-" * 72)
    print("Q > 0.707 时通带内出现峰化，Q 越大越接近自激振荡（Q -> inf 即振荡器）。")
    print()

    freqs = log_sweep(10, 1e5, points_per_decade=3)
    gains = [db(f.transfer(x)) for x in freqs]
    ascii_bode(freqs, gains, title="Sallen-Key 二阶低通 ASCII 波特图")
    print()


# =================================================================
# 4. 运放增益带宽积
# =================================================================
def demo_gbw():
    print("=" * 72)
    print("【4】运放增益带宽积 GBW 与闭环带宽")
    print("=" * 72)
    a0 = 1e5           # 开环直流增益 100 dB
    fb = 10.0          # 开环第一极点（主极点）
    gbw = a0 * fb
    print(f"运放参数: 开环增益 A0 = {a0:.0e} ({db(a0):.1f} dB), "
          f"主极点 fb = {fb:.1f} Hz")
    print(f"增益带宽积 GBW = A0 * fb = {gbw/1e6:.3f} MHz")
    print("-" * 72)
    print(f"{'闭环增益':>10} | {'dB':>8} | {'闭环带宽 (kHz)':>16} | {'GBW 校验 (MHz)':>16}")
    print("-" * 72)
    for acl in [1, 2, 10, 20, 100, 1000]:
        bw = gbw / acl
        print(f"{acl:10d} | {db(acl):8.2f} | {bw/1e3:16.2f} | {acl*bw/1e6:16.3f}")
    print("-" * 72)
    print("结论：闭环增益 x 闭环带宽 = 常数。要高增益又要大带宽，只能多级级联。")
    print()

    print("--- 多级级联的带宽收缩 ---")
    print(f"{'级数 n':>8} | {'每级增益':>10} | {'每级带宽 kHz':>14} | {'总带宽 kHz':>13}")
    print("-" * 72)
    total_gain = 1000
    for n in [1, 2, 3, 4]:
        per_gain = total_gain ** (1.0 / n)
        per_bw = gbw / per_gain
        # n 级相同一阶级联，总带宽 = per_bw * sqrt(2^(1/n) - 1)
        shrink = math.sqrt(2 ** (1.0 / n) - 1)
        total_bw = per_bw * shrink
        print(f"{n:8d} | {per_gain:10.2f} | {per_bw/1e3:14.2f} | {total_bw/1e3:13.2f}")
    print("-" * 72)
    print("总增益 1000 倍时，用 3 级比 1 级带宽宽得多 —— 多级放大的价值所在。")
    print()


def main():
    demo_rc_filter()
    demo_amplifier_response()
    demo_active_filter()
    demo_gbw()
    print("=" * 72)
    print("要点回顾：")
    print("  1) 一阶极点 -> -20dB/十倍频 + 最大 -90 度相移")
    print("  2) 米勒效应把 Cbc 放大 (1+|Av|) 倍，是共射级带宽的主要限制")
    print("  3) 有源滤波器用运放实现无电感的高阶滤波，Q 决定通带平坦度")
    print("  4) 运放闭环: 增益 x 带宽 = GBW = 常数")
    print("=" * 72)


if __name__ == "__main__":
    main()
