"""
集成运放电路计算器与负反馈分析
====================================

运放是模拟电路的"通用积木"。本程序覆盖：

    1. 理想运放线性应用：反相/同相/加法/减法/积分/微分 —— 虚短虚断法则
    2. 有限增益修正：真实运放与理想模型的误差有多大
    3. 负反馈四种组态的定量对比（增益、灵敏度、输入输出阻抗）
    4. 反馈深度对增益稳定性的影响（灵敏度公式 dAf/Af = 1/(1+AF) * dA/A）
    5. 环路稳定性：相位裕度与自激振荡判据
    6. 迟滞比较器（施密特触发器）阈值计算与抗噪声演示

仅使用标准库 math / cmath，直接运行：
    python opamp_sim.py
"""

import cmath
import math


# =================================================================
# 1. 理想运放线性应用
# =================================================================
class IdealOpAmp:
    """
    理想运放两大法则（工作在线性区/负反馈条件下）：
        虚短 (Virtual Short) : V+ = V-    因为 Aod -> 无穷，Vo 有限 => Vid = 0
        虚断 (Virtual Open)  : I+ = I- = 0 因为 Rid -> 无穷
    注意：虚短虚断只在负反馈闭环时成立，开环或正反馈时失效！
    """

    @staticmethod
    def inverting(vin, rf, r1):
        """反相放大器: Vo = -(Rf/R1)*Vin,  Rin = R1"""
        av = -rf / r1
        return {"Av": av, "Vo": av * vin, "Rin": r1,
                "name": "反相放大", "formula": "Av = -Rf/R1"}

    @staticmethod
    def non_inverting(vin, rf, r1):
        """同相放大器: Vo = (1 + Rf/R1)*Vin,  Rin -> 无穷大"""
        av = 1 + rf / r1
        return {"Av": av, "Vo": av * vin, "Rin": float("inf"),
                "name": "同相放大", "formula": "Av = 1 + Rf/R1"}

    @staticmethod
    def voltage_follower(vin):
        """电压跟随器: Vo = Vin，增益 1，纯做阻抗变换。"""
        return {"Av": 1.0, "Vo": vin, "Rin": float("inf"),
                "name": "电压跟随器", "formula": "Av = 1"}

    @staticmethod
    def summing(vins, rs, rf):
        """反相加法器: Vo = -Rf*(V1/R1 + V2/R2 + ...)"""
        vo = -rf * sum(v / r for v, r in zip(vins, rs))
        return {"Vo": vo, "name": "反相加法器", "formula": "Vo = -Rf*sum(Vi/Ri)"}

    @staticmethod
    def difference(v1, v2, r1, r2, r3, r4):
        """
        差分放大器（减法器）:
            Vo = (R4/(R3+R4))*(1+R2/R1)*V2 - (R2/R1)*V1
        当 R1=R3, R2=R4 时简化为 Vo = (R2/R1)*(V2 - V1)
        """
        vo = (r4 / (r3 + r4)) * (1 + r2 / r1) * v2 - (r2 / r1) * v1
        return {"Vo": vo, "name": "差分放大器",
                "formula": "Vo = (R2/R1)*(V2-V1)  [R1=R3,R2=R4 时]"}

    @staticmethod
    def integrator(vin_func, r, c, t_end, dt, vo0=0.0):
        """
        积分器: Vo(t) = -(1/(R*C)) * integral(Vin dt) + Vo(0)
        用梯形法数值积分，返回 (t, vin, vo) 采样列表。
        """
        samples = []
        vo = vo0
        t = 0.0
        n = int(t_end / dt)
        for i in range(n + 1):
            vin = vin_func(t)
            samples.append((t, vin, vo))
            vin_next = vin_func(t + dt)
            vo += -(1.0 / (r * c)) * 0.5 * (vin + vin_next) * dt
            t += dt
        return samples

    @staticmethod
    def differentiator(vin_func, r, c, t_end, dt):
        """微分器: Vo(t) = -R*C * dVin/dt，用中心差分。"""
        samples = []
        t = 0.0
        n = int(t_end / dt)
        for i in range(n + 1):
            dv = (vin_func(t + dt) - vin_func(max(t - dt, 0.0))) / (2 * dt)
            samples.append((t, vin_func(t), -r * c * dv))
            t += dt
        return samples


def demo_linear_apps():
    print("=" * 72)
    print("【1】理想运放线性应用（虚短虚断法则）")
    print("=" * 72)
    op = IdealOpAmp()
    vin = 0.1
    results = [
        op.inverting(vin, rf=100e3, r1=10e3),
        op.non_inverting(vin, rf=100e3, r1=10e3),
        op.voltage_follower(vin),
    ]
    print(f"输入 Vin = {vin} V,  Rf = 100k, R1 = 10k")
    print("-" * 72)
    print(f"{'电路':<14} | {'公式':<22} | {'Av':>9} | {'Vo (V)':>9} | {'Rin':>10}")
    print("-" * 72)
    for r in results:
        rin = "inf" if r["Rin"] == float("inf") else f"{r['Rin']/1e3:.0f}k"
        print(f"{r['name']:<14} | {r['formula']:<22} | {r['Av']:9.3f} | "
              f"{r['Vo']:9.4f} | {rin:>10}")
    print("-" * 72)
    print("反相放大器的 '虚地' 是关键：V- = V+ = 0，所以 Rin 就等于 R1。")
    print()

    print("--- 反相加法器（三路混音 / DAC 的原型）---")
    vins = [0.5, -0.2, 0.3]
    rs = [10e3, 20e3, 40e3]
    rf = 20e3
    s = op.summing(vins, rs, rf)
    print(f"输入: V1={vins[0]}V (R1=10k), V2={vins[1]}V (R2=20k), V3={vins[2]}V (R3=40k), Rf=20k")
    for i, (v, r) in enumerate(zip(vins, rs), 1):
        print(f"  第{i}路贡献: -Rf/R{i} * V{i} = {-rf/r:.2f} * {v} = {-rf/r*v:+.4f} V")
    print(f"  合计 Vo = {s['Vo']:.4f} V")
    print("  权重 = Rf/Ri，按 1:1/2:1/4 取电阻即得二进制加权 DAC。")
    print()

    print("--- 差分放大器（传感器桥式信号提取）---")
    d = op.difference(v1=2.005, v2=2.010, r1=1e3, r2=100e3, r3=1e3, r4=100e3)
    print(f"输入: V1 = 2.005 V, V2 = 2.010 V (共模 2.0075 V, 差模 5 mV)")
    print(f"R1=R3=1k, R2=R4=100k  ->  差模增益 = 100")
    print(f"输出 Vo = {d['Vo']:.6f} V   (理论 100 * 5mV = 0.5 V)")
    print("共模信号被完全抵消 —— 这正是从大共模里提取小差模的方法。")
    # 电阻失配导致的 CMRR 退化
    print()
    print("电阻失配对共模抑制比 CMRR 的影响:")
    print(f"{'失配度':>8} | {'共模增益 Acm':>14} | {'CMRR (dB)':>11}")
    print("-" * 72)
    for tol in [0.0001, 0.001, 0.01, 0.05]:
        r1, r2, r3 = 1e3, 100e3, 1e3
        r4 = 100e3 * (1 + tol)
        vo_cm = op.difference(1.0, 1.0, r1, r2, r3, r4)["Vo"]
        acm = abs(vo_cm / 1.0)
        cmrr = 20 * math.log10(100 / acm) if acm > 1e-15 else 999
        print(f"{tol*100:7.2f}% | {acm:14.6f} | {cmrr:11.2f}")
    print("-" * 72)
    print("结论：CMRR 由电阻匹配精度决定，这就是仪表放大器要用集成电阻网络的原因。")
    print()

    print("--- 积分器：方波输入 -> 三角波输出 ---")
    r, c = 10e3, 100e-9
    tau = r * c
    freq = 500.0

    def square(t):
        return 1.0 if math.sin(2 * math.pi * freq * t) >= 0 else -1.0

    samples = op.integrator(square, r, c, t_end=2.0 / freq, dt=1.0 / (freq * 400))
    print(f"R = {r/1e3:.0f}k, C = {c*1e9:.0f}nF, RC = {tau*1e3:.3f} ms, "
          f"输入 {freq:.0f}Hz 方波 (+/-1V)")
    print(f"理论峰峰值 Vpp = Vin*T/(2*R*C) = {1.0/(freq*2*r*c):.4f} V")
    print(f"{'t (ms)':>9} | {'Vin (V)':>8} | {'Vo (V)':>9}")
    print("-" * 40)
    step = max(1, len(samples) // 14)
    for i in range(0, len(samples), step):
        t, vi, vo = samples[i]
        print(f"{t*1e3:9.4f} | {vi:8.2f} | {vo:9.5f}")
    vo_list = [s[2] for s in samples]
    print("-" * 40)
    print(f"实测峰峰值 = {max(vo_list) - min(vo_list):.4f} V   "
          f"（输入为常数时输出是斜坡，斜率 = -Vin/RC）")
    print()


# =================================================================
# 2. 有限开环增益的修正
# =================================================================
def finite_gain_analysis():
    print("=" * 72)
    print("【2】理想模型 vs 有限开环增益：误差到底有多大")
    print("=" * 72)
    print("同相放大器精确式:  Af = A / (1 + A*F),  其中 F = R1/(R1+Rf)")
    r1, rf = 10e3, 90e3
    f_fb = r1 / (r1 + rf)          # 反馈系数
    ideal = 1 + rf / r1            # 理想闭环增益 = 1/F = 10
    print(f"R1 = {r1/1e3:.0f}k, Rf = {rf/1e3:.0f}k  ->  F = {f_fb:.4f}, "
          f"理想闭环增益 1/F = {ideal:.2f}")
    print("-" * 72)
    print(f"{'开环增益 A':>12} | {'A dB':>8} | {'环路增益 AF':>12} | "
          f"{'实际 Af':>10} | {'相对误差':>10}")
    print("-" * 72)
    for a in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:
        af = a / (1 + a * f_fb)
        err = (af - ideal) / ideal * 100
        print(f"{a:12.0e} | {20*math.log10(a):8.1f} | {a*f_fb:12.1f} | "
              f"{af:10.5f} | {err:9.4f}%")
    print("-" * 72)
    print("结论：只要环路增益 AF >> 1，闭环增益就只由外部电阻决定，与运放本身无关。")
    print("      这就是负反馈最伟大的地方：用精密无源器件锁定不精密有源器件的性能。")
    print()


# =================================================================
# 3. 负反馈四种组态
# =================================================================
FEEDBACK_TOPOLOGIES = {
    "电压串联": {
        "sample": "电压", "mix": "串联",
        "A_unit": "V/V (Av)", "stabilize": "电压增益",
        "Rin": "增大 (1+AF) 倍", "Rout": "减小 (1+AF) 倍",
        "use": "同相放大器 / 电压跟随器 / 理想电压放大",
    },
    "电压并联": {
        "sample": "电压", "mix": "并联",
        "A_unit": "V/A (Rm 互阻)", "stabilize": "互阻增益",
        "Rin": "减小 (1+AF) 倍", "Rout": "减小 (1+AF) 倍",
        "use": "反相放大器 / 光电二极管跨阻放大 TIA",
    },
    "电流串联": {
        "sample": "电流", "mix": "串联",
        "A_unit": "A/V (Gm 互导)", "stabilize": "互导增益",
        "Rin": "增大 (1+AF) 倍", "Rout": "增大 (1+AF) 倍",
        "use": "压控电流源 / LED 恒流驱动",
    },
    "电流并联": {
        "sample": "电流", "mix": "并联",
        "A_unit": "A/A (Ai)", "stabilize": "电流增益",
        "Rin": "减小 (1+AF) 倍", "Rout": "增大 (1+AF) 倍",
        "use": "电流放大 / 恒流源",
    },
}


def demo_feedback_topologies():
    print("=" * 72)
    print("【3】负反馈四种组态速查表")
    print("=" * 72)
    print("判别口诀：")
    print("  取样：反馈网络并接在输出端 -> 电压取样；串接在输出回路 -> 电流取样")
    print("  比较：反馈信号与输入信号加在不同电极 -> 串联；同一电极 -> 并联")
    print("  记忆：取样什么就稳定什么；串联提高输入阻抗，并联降低输入阻抗；")
    print("        电压取样降低输出阻抗，电流取样提高输出阻抗。")
    print()
    print(f"{'组态':<10} | {'稳定量':<10} | {'输入阻抗':<14} | {'输出阻抗':<14}")
    print("-" * 72)
    for name, t in FEEDBACK_TOPOLOGIES.items():
        print(f"{name:<10} | {t['stabilize']:<10} | {t['Rin']:<14} | {t['Rout']:<14}")
    print("-" * 72)
    for name, t in FEEDBACK_TOPOLOGIES.items():
        print(f"  {name}: {t['use']}")
    print()

    # 定量演示
    print("--- 定量演示：电压串联负反馈对各项指标的改善 ---")
    a_open, rin_open, rout_open = 1e4, 10e3, 1e3
    print(f"开环: A = {a_open:.0e}, Rin = {rin_open/1e3:.0f}k, Rout = {rout_open:.0f} Ohm")
    print(f"{'F':>10} | {'1+AF':>10} | {'Af':>10} | {'Rif (MOhm)':>12} | {'Rof (Ohm)':>11}")
    print("-" * 72)
    for f_fb in [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]:
        d = 1 + a_open * f_fb
        af = a_open / d
        rif = rin_open * d
        rof = rout_open / d
        print(f"{f_fb:10.3f} | {d:10.1f} | {af:10.3f} | {rif/1e6:12.3f} | {rof:11.5f}")
    print("-" * 72)
    print("反馈深度 D = 1+AF 是核心参数：增益降 D 倍，换来各项指标改善 D 倍。")
    print()


def demo_sensitivity():
    print("=" * 72)
    print("【4】增益稳定性：负反馈的灵敏度改善")
    print("=" * 72)
    print("灵敏度公式:  dAf/Af = [1/(1+AF)] * (dA/A)")
    print("含义：开环增益变化 10%，闭环只变化 10%/(1+AF)")
    print("-" * 72)
    a_nom = 1e4
    delta = 0.30            # 开环增益变化 30%（温度/器件离散）
    print(f"设开环增益 A 因温度/离散变化 {delta*100:.0f}%")
    print(f"{'F':>8} | {'1+AF':>9} | {'Af(标称)':>10} | {'Af(变化后)':>11} | "
          f"{'闭环变化率':>11} | {'改善倍数':>9}")
    print("-" * 72)
    for f_fb in [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]:
        d = 1 + a_nom * f_fb
        af0 = a_nom / d
        a2 = a_nom * (1 - delta)
        af1 = a2 / (1 + a2 * f_fb)
        change = abs(af1 - af0) / af0 * 100
        improve = (delta * 100) / change if change > 1e-9 else float("inf")
        imp_str = f"{improve:9.1f}" if improve != float("inf") else "      inf"
        print(f"{f_fb:8.3f} | {d:9.1f} | {af0:10.3f} | {af1:11.3f} | "
              f"{change:10.4f}% | {imp_str}")
    print("-" * 72)
    print("同理，非线性失真、噪声（反馈环内的）、频率响应都被改善 (1+AF) 倍。")
    print()


# =================================================================
# 4. 环路稳定性
# =================================================================
def loop_gain(f, a0=1e5, poles=(10.0, 1e5, 1e6)):
    """三极点运放的开环传递函数 A(jf)。"""
    h = complex(a0, 0)
    for p in poles:
        h /= (1 + 1j * f / p)
    return h


def unwrapped_phase(f, a0=1e5, poles=(10.0, 1e5, 1e6)):
    """
    累加各极点相移得到连续（不缠绕）的相位，单位度。
    每个一阶极点贡献 -arctan(f/p)，最大 -90 度。
    """
    return -sum(math.degrees(math.atan2(f, p)) for p in poles)


def demo_stability():
    print("=" * 72)
    print("【5】环路稳定性：相位裕度与自激振荡")
    print("=" * 72)
    print("自激振荡条件（巴克豪森判据）:  |AF| = 1  且  相移 = -180 度")
    print("稳定判据：在 |AF| = 1 (0 dB) 处，相位裕度 PM = 180 + phase > 45 度为安全")
    print("-" * 72)
    poles = (10.0, 1e5, 1e6)
    a0 = 1e5
    print(f"运放模型: A0 = {a0:.0e}, 三个极点 = {poles[0]:.0f} Hz, "
          f"{poles[1]:.0e} Hz, {poles[2]:.0e} Hz")
    print()
    print(f"{'F':>8} | {'闭环增益':>10} | {'穿越频率 fc (Hz)':>18} | "
          f"{'相位裕度 (deg)':>16} | {'判定':<12}")
    print("-" * 78)
    for f_fb in [1.0, 0.1, 0.01, 0.001]:
        # 找 |A*F| = 1 的频率（二分法）
        lo, hi = 1.0, 1e8
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if abs(loop_gain(mid, a0, poles) * f_fb) > 1.0:
                lo = mid
            else:
                hi = mid
        fc = math.sqrt(lo * hi)
        ph = unwrapped_phase(fc, a0, poles)
        pm = 180 + ph
        if pm > 60:
            verdict = "很稳定"
        elif pm > 45:
            verdict = "稳定"
        elif pm > 0:
            verdict = "临界/振铃"
        else:
            verdict = "自激振荡!"
        print(f"{f_fb:8.3f} | {1/f_fb:10.1f} | {fc:18.2f} | {pm:16.2f} | {verdict:<12}")
    print("-" * 78)
    print("规律：反馈越深（F 越大、闭环增益越低），穿越频率越高，相位裕度越小。")
    print("      电压跟随器 (F=1) 是最苛刻的情况 —— 所以要用'单位增益稳定'的运放。")
    print()
    print("消除自激的常用手段：")
    print("  1) 主极点补偿：加大电容把第一个极点推向低频（牺牲带宽换稳定）")
    print("  2) 密勒补偿：利用米勒效应用小电容实现大等效电容（集成电路首选）")
    print("  3) 超前补偿：在反馈网络并联小电容，抵消一部分相移")
    print()

    print("--- 开环增益的波特图（含相位）---")
    print(f"{'f (Hz)':>12} | {'|A| dB':>9} | {'相位 (deg)':>12}")
    print("-" * 42)
    for f in [1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7]:
        h = loop_gain(f, a0, poles)
        print(f"{f:12.0f} | {20*math.log10(abs(h)):9.2f} | "
              f"{unwrapped_phase(f, a0, poles):12.2f}")
    print("-" * 42)
    print("相位从 0 度单调走向 -270 度（三个极点各贡献 -90 度）。")
    print()


# =================================================================
# 5. 迟滞比较器（施密特触发器）
# =================================================================
class SchmittTrigger:
    """
    同相输入迟滞比较器（正反馈）：

        上门限 VTH+ = Vref*(1+R1/R2) - Vol*(R1/R2)
        下门限 VTH- = Vref*(1+R1/R2) - Voh*(R1/R2)
        回差   dV   = (Voh - Vol) * R1/R2

    注意：这里是正反馈，虚短虚断法则不成立！
    """

    def __init__(self, r1=10e3, r2=100e3, voh=5.0, vol=-5.0, vref=0.0):
        self.r1, self.r2 = r1, r2
        self.voh, self.vol, self.vref = voh, vol, vref
        k = r1 / r2
        self.vth_hi = vref * (1 + k) - vol * k
        self.vth_lo = vref * (1 + k) - voh * k
        self.hysteresis = (voh - vol) * k
        self.state = vol

    def step(self, vin):
        if self.state == self.voh:
            if vin < self.vth_lo:
                self.state = self.vol
        else:
            if vin > self.vth_hi:
                self.state = self.voh
        return self.state


def demo_schmitt():
    print("=" * 72)
    print("【6】迟滞比较器（施密特触发器）—— 运放的非线性应用")
    print("=" * 72)
    st = SchmittTrigger(r1=10e3, r2=100e3, voh=5.0, vol=-5.0)
    print(f"R1 = {st.r1/1e3:.0f}k, R2 = {st.r2/1e3:.0f}k, "
          f"输出摆幅 {st.vol} ~ {st.voh} V")
    print(f"上门限 VTH+ = {st.vth_hi:.4f} V")
    print(f"下门限 VTH- = {st.vth_lo:.4f} V")
    print(f"回差电压 dV = {st.hysteresis:.4f} V")
    print()
    print("--- 带噪声的缓变输入信号（对比普通比较器）---")
    print(f"{'t':>4} | {'Vin (V)':>9} | {'普通比较器':>10} | {'施密特':>8}")
    print("-" * 46)
    # 伪随机噪声（线性同余，避免引入 random 依赖以保证可复现）
    seed = 12345

    def noise():
        nonlocal seed
        seed = (1103515245 * seed + 12345) % (2 ** 31)
        return (seed / (2 ** 31) - 0.5) * 0.9     # +/-0.45 V 噪声（小于回差 1.0 V）

    plain_flips, schmitt_flips = 0, 0
    plain_prev, sch_prev = None, None
    for i in range(60):
        clean = -1.2 + 0.04 * i           # 从 -1.2 V 缓慢升到 +1.16 V
        vin = clean + noise()
        plain = 5.0 if vin > 0 else -5.0
        sch = st.step(vin)
        if plain_prev is not None and plain != plain_prev:
            plain_flips += 1
        if sch_prev is not None and sch != sch_prev:
            schmitt_flips += 1
        plain_prev, sch_prev = plain, sch
        if 22 <= i <= 42:
            mark = "  <-- 误翻转" if (plain_prev == plain and i > 22 and
                                    plain != sch and sch == st.vol) else ""
            print(f"{i:4d} | {vin:9.4f} | {plain:10.1f} | {sch:8.1f}{mark}")
    print("-" * 46)
    print(f"输出翻转次数：普通比较器 {plain_flips} 次，施密特触发器 {schmitt_flips} 次")
    print("结论：回差 > 噪声峰峰值时，输出干净无抖动。这是所有数字输入口的标配。")
    print()


def main():
    demo_linear_apps()
    finite_gain_analysis()
    demo_feedback_topologies()
    demo_sensitivity()
    demo_stability()
    demo_schmitt()
    print("=" * 72)
    print("要点回顾：")
    print("  1) 线性区用虚短虚断，非线性区（比较器/正反馈）绝对不能用")
    print("  2) 环路增益 AF >> 1 时，闭环特性只由外部无源网络决定")
    print("  3) 反馈深度 D = 1+AF：增益换性能，是模拟设计的核心交易")
    print("  4) 深度负反馈 + 多极点 = 自激振荡，必须做频率补偿")
    print("=" * 72)


if __name__ == "__main__":
    main()
