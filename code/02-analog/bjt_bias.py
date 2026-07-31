"""
BJT 与 MOSFET 静态工作点 / 小信号参数计算器
==============================================

模拟电路设计的两步走范式：
    第一步（直流分析）：算静态工作点 Q(ICQ, VCEQ)，决定管子工作在哪个区
    第二步（交流分析）：在 Q 点线性化，得到小信号模型，算增益/输入输出电阻

本程序实现：
    1. BJT 分压式偏置电路（射极负反馈偏置）的 Q 点求解
    2. 温度/beta 漂移对 Q 点的稳定性分析（分压偏置 vs 固定偏置）
    3. 共射(CE) / 共集(CC) / 共基(CB) 三种组态的小信号指标对比
    4. MOSFET 共源放大器的 Q 点与增益计算

仅使用标准库，直接运行：
    python bjt_bias.py
"""

import math

VT = 0.02585        # 热电压 @ 300K，约 25.85 mV
VBE_ON = 0.7        # 硅管发射结导通压降近似值
VBE_TC = -2.2e-3    # VBE 温度系数 -2.2 mV/C


# =================================================================
# 1. BJT 分压式偏置
# =================================================================
class BJTAmplifier:
    """
    典型分压式偏置共射放大电路:

        VCC ----+-------+------- Rc ----+---- (集电极 -> C2 -> RL)
                |       |               |
               R1       |             [ C ]
                |       +------------ B  BJT (NPN)
                +--- B  |             [ E ]
                |       |               |
               R2      C1(旁路)         Re
                |       |               |
        GND ----+-------+---------------+
    """

    def __init__(self, vcc=12.0, r1=40e3, r2=10e3, rc=3.3e3, re=1.0e3,
                 beta=100.0, rl=None, re_unbypassed=0.0, va=100.0):
        self.vcc = vcc
        self.r1 = r1
        self.r2 = r2
        self.rc = rc
        self.re = re                     # 直流射极电阻（总）
        self.re_ub = re_unbypassed       # 交流未被旁路的射极电阻
        self.beta = beta
        self.rl = rl
        self.va = va                     # 厄尔利电压，用于算 ro

    # ---------- 直流分析 ----------
    def dc_operating_point(self, temp_c=27.0):
        """戴维南等效后求 Q 点，返回 dict。"""
        vbe = VBE_ON + VBE_TC * (temp_c - 27.0)
        vth = self.vcc * self.r2 / (self.r1 + self.r2)   # 戴维南等效电压
        rth = self.r1 * self.r2 / (self.r1 + self.r2)    # 戴维南等效内阻

        # 基极回路 KVL: Vth = Ib*Rth + Vbe + (beta+1)*Ib*Re
        ib = (vth - vbe) / (rth + (self.beta + 1) * self.re)
        ib = max(ib, 0.0)
        ic = self.beta * ib
        ie = (self.beta + 1) * ib
        vce = self.vcc - ic * self.rc - ie * self.re
        ve = ie * self.re
        vc = self.vcc - ic * self.rc
        vb = ve + vbe

        # 判断工作区
        if vce > 0.3 and ib > 0:
            region = "放大区 (Active)"
        elif ib <= 0:
            region = "截止区 (Cutoff)"
        else:
            region = "饱和区 (Saturation) —— 需减小 Rc 或降低 Q 点!"

        return {
            "Vth": vth, "Rth": rth, "Vbe": vbe,
            "IB": ib, "ICQ": ic, "IE": ie,
            "VCEQ": vce, "VB": vb, "VE": ve, "VC": vc,
            "region": region, "temp": temp_c,
        }

    # ---------- 小信号参数 ----------
    def small_signal_params(self, temp_c=27.0):
        q = self.dc_operating_point(temp_c)
        ic = q["ICQ"]
        if ic <= 0:
            return None
        gm = ic / VT                       # 跨导 gm = IC/VT
        r_be = (self.beta + 1) * VT / (q["IE"]) if q["IE"] > 0 else float("inf")
        r_pi = self.beta / gm              # r_pi = beta/gm，与 r_be 基本等价
        ro = self.va / ic                  # 输出电阻（厄尔利效应）
        return {"gm": gm, "rbe": r_be, "rpi": r_pi, "ro": ro, "Q": q}

    # ---------- 三种组态增益 ----------
    def gain_common_emitter(self, temp_c=27.0):
        """共射：反相电压放大，Av = -gm*(Rc//RL) / (1 + gm*Re_ub)"""
        p = self.small_signal_params(temp_c)
        if p is None:
            return None
        rc_eff = self._parallel(self.rc, self.rl)
        gm, rbe = p["gm"], p["rbe"]
        av = -self.beta * rc_eff / (rbe + (self.beta + 1) * self.re_ub)
        rin_base = rbe + (self.beta + 1) * self.re_ub
        rin = self._parallel(self._parallel(self.r1, self.r2), rin_base)
        rout = self.rc
        return {"Av": av, "Rin": rin, "Rout": rout, "type": "共射 CE"}

    def gain_common_collector(self, temp_c=27.0):
        """共集（射极跟随器）：Av ~ +1，高输入阻抗、低输出阻抗。"""
        p = self.small_signal_params(temp_c)
        if p is None:
            return None
        rbe = p["rbe"]
        re_eff = self._parallel(self.re, self.rl)
        av = (self.beta + 1) * re_eff / (rbe + (self.beta + 1) * re_eff)
        rin_base = rbe + (self.beta + 1) * re_eff
        rin = self._parallel(self._parallel(self.r1, self.r2), rin_base)
        rs = self._parallel(self.r1, self.r2)     # 信号源看到的基极侧电阻
        rout = self._parallel(self.re, (rbe + rs) / (self.beta + 1))
        return {"Av": av, "Rin": rin, "Rout": rout, "type": "共集 CC"}

    def gain_common_base(self, temp_c=27.0):
        """共基：同相放大，输入阻抗极低（约 rbe/(1+beta)），高频特性最好。"""
        p = self.small_signal_params(temp_c)
        if p is None:
            return None
        rbe = p["rbe"]
        rc_eff = self._parallel(self.rc, self.rl)
        av = self.beta * rc_eff / rbe
        rin = self._parallel(self.re, rbe / (self.beta + 1))
        rout = self.rc
        return {"Av": av, "Rin": rin, "Rout": rout, "type": "共基 CB"}

    @staticmethod
    def _parallel(a, b):
        if b is None:
            return a
        if a is None:
            return b
        return a * b / (a + b)


def demo_bjt_q_point():
    print("=" * 70)
    print("【1】BJT 分压式偏置电路静态工作点")
    print("=" * 70)
    amp = BJTAmplifier(vcc=12.0, r1=40e3, r2=10e3, rc=3.3e3, re=1.0e3, beta=100)
    q = amp.dc_operating_point()
    print(f"电路参数: VCC={amp.vcc}V  R1={amp.r1/1e3:.0f}k  R2={amp.r2/1e3:.0f}k  "
          f"Rc={amp.rc/1e3:.1f}k  Re={amp.re/1e3:.1f}k  beta={amp.beta:.0f}")
    print("-" * 70)
    print(f"  戴维南等效   : Vth = {q['Vth']:.4f} V,  Rth = {q['Rth']/1e3:.3f} kOhm")
    print(f"  基极电流     : IB   = {q['IB']*1e6:.3f} uA")
    print(f"  集电极电流   : ICQ  = {q['ICQ']*1e3:.4f} mA")
    print(f"  管压降       : VCEQ = {q['VCEQ']:.4f} V")
    print(f"  各点电位     : VB = {q['VB']:.3f} V, VE = {q['VE']:.3f} V, VC = {q['VC']:.3f} V")
    print(f"  工作区判定   : {q['region']}")
    print()
    print("经验法则：VCEQ 取 VCC 的 1/3~1/2，可获得最大不失真输出摆幅。")
    print(f"  本例 VCEQ/VCC = {q['VCEQ']/amp.vcc:.2f}  ->  "
          f"{'合理' if 0.25 < q['VCEQ']/amp.vcc < 0.6 else '需调整'}")
    print()
    return amp


# =================================================================
# 2. Q 点稳定性：分压偏置 vs 固定偏置
# =================================================================
def demo_stability():
    print("=" * 70)
    print("【2】Q 点稳定性对比：beta 离散 与 温度漂移")
    print("=" * 70)

    print("--- (a) beta 从 50 变到 300（同批次晶体管的典型离散度）---")
    print(f"{'beta':>6} | {'分压偏置 ICQ(mA)':>18} | {'固定偏置 ICQ(mA)':>18}")
    print("-" * 70)
    icq_div, icq_fix = [], []
    for beta in [50, 100, 150, 200, 300]:
        a1 = BJTAmplifier(r1=40e3, r2=10e3, rc=3.3e3, re=1.0e3, beta=beta)
        # 固定偏置：仅一只基极电阻 Rb，无 Re
        a2 = BJTAmplifier(r1=470e3, r2=1e12, rc=3.3e3, re=1e-9, beta=beta)
        i1 = a1.dc_operating_point()["ICQ"] * 1e3
        i2 = a2.dc_operating_point()["ICQ"] * 1e3
        icq_div.append(i1)
        icq_fix.append(i2)
        print(f"{beta:6d} | {i1:18.4f} | {i2:18.4f}")
    print("-" * 70)
    sp_div = (max(icq_div) - min(icq_div)) / (sum(icq_div) / len(icq_div)) * 100
    sp_fix = (max(icq_fix) - min(icq_fix)) / (sum(icq_fix) / len(icq_fix)) * 100
    print(f"ICQ 相对变化范围:  分压偏置 {sp_div:.1f}%   固定偏置 {sp_fix:.1f}%")
    print("结论：射极电阻 Re 引入直流负反馈，把 ICQ 牢牢\"钉\"在 (Vth-Vbe)/Re 上。")
    print()

    print("--- (b) 温度从 -20C 升到 80C（VBE 温漂 -2.2 mV/C）---")
    print(f"{'T (C)':>7} | {'Vbe (V)':>9} | {'ICQ (mA)':>10} | {'VCEQ (V)':>10}")
    print("-" * 70)
    amp = BJTAmplifier(r1=40e3, r2=10e3, rc=3.3e3, re=1.0e3, beta=100)
    for t in [-20, 0, 27, 50, 80]:
        q = amp.dc_operating_point(temp_c=t)
        print(f"{t:7d} | {q['Vbe']:9.4f} | {q['ICQ']*1e3:10.4f} | {q['VCEQ']:10.4f}")
    print("-" * 70)
    print("结论：Re 越大温漂越小，但会牺牲输出摆幅 —— 典型的设计折中。")
    print()


# =================================================================
# 3. 三种组态对比
# =================================================================
def demo_configurations():
    print("=" * 70)
    print("【3】共射 / 共集 / 共基 三种组态小信号指标对比")
    print("=" * 70)
    amp = BJTAmplifier(vcc=12.0, r1=40e3, r2=10e3, rc=3.3e3, re=1.0e3,
                       beta=100, rl=10e3)
    p = amp.small_signal_params()
    print(f"Q 点小信号参数: gm = {p['gm']*1e3:.3f} mS,  rbe = {p['rbe']:.1f} Ohm,  "
          f"ro = {p['ro']/1e3:.1f} kOhm")
    print("-" * 70)
    print(f"{'组态':<10} | {'Av':>12} | {'|Av| dB':>9} | {'Rin':>12} | {'Rout':>11}")
    print("-" * 70)
    for fn in (amp.gain_common_emitter, amp.gain_common_collector, amp.gain_common_base):
        g = fn()
        db = 20 * math.log10(abs(g["Av"])) if abs(g["Av"]) > 0 else -999
        print(f"{g['type']:<10} | {g['Av']:12.3f} | {db:9.2f} | "
              f"{g['Rin']:12.1f} | {g['Rout']:11.1f}")
    print("-" * 70)
    print("选型口诀：")
    print("  共射 —— 电压增益大且反相，做主放大级")
    print("  共集 —— 电压增益约 1，输入阻抗高、输出阻抗低，做缓冲/输出级")
    print("  共基 —— 无米勒效应，高频响应最好，做射频前端/宽带级")
    print()

    # 射极电阻部分不旁路，交流负反馈牺牲增益换线性度
    print("--- 射极电阻部分不旁路（交流负反馈）对增益的影响 ---")
    print(f"{'Re_ub (Ohm)':>12} | {'Av':>10} | {'Rin (kOhm)':>12} | {'增益下降':>10}")
    print("-" * 70)
    av0 = None
    for re_ub in [0, 20, 50, 100, 200, 500]:
        a = BJTAmplifier(vcc=12.0, r1=40e3, r2=10e3, rc=3.3e3, re=1.0e3,
                         beta=100, rl=10e3, re_unbypassed=re_ub)
        g = a.gain_common_emitter()
        if av0 is None:
            av0 = abs(g["Av"])
        drop = abs(g["Av"]) / av0
        print(f"{re_ub:12d} | {g['Av']:10.2f} | {g['Rin']/1e3:12.2f} | {drop:10.3f}")
    print("-" * 70)
    print("结论：牺牲增益换取更好的线性度、更稳定的增益、更高的输入阻抗。")
    print()


# =================================================================
# 4. MOSFET 共源放大器
# =================================================================
class MOSFETAmplifier:
    """
    N 沟道增强型 MOS 共源放大电路（分压偏置 + 源极电阻）

    饱和区电流方程:  ID = 0.5 * k_n * (VGS - Vth)^2 * (1 + lambda*VDS)
    其中 k_n = un*Cox*(W/L)，单位 A/V^2
    """

    def __init__(self, vdd=12.0, r1=2e6, r2=1e6, rd=4.7e3, rs=1e3,
                 kn=2e-3, vth=1.5, lam=0.01, rl=None):
        self.vdd, self.r1, self.r2 = vdd, r1, r2
        self.rd, self.rs = rd, rs
        self.kn, self.vth, self.lam = kn, vth, lam
        self.rl = rl

    def dc_operating_point(self):
        """
        栅极无电流 -> VG 由分压直接确定。
        解方程: VG - VGS - ID*Rs = 0，且 ID = 0.5*kn*(VGS-Vth)^2
        代入得关于 VGS 的一元二次方程。
        """
        vg = self.vdd * self.r2 / (self.r1 + self.r2)
        # 0.5*kn*Rs*(VGS-Vth)^2 + VGS - VG = 0
        a = 0.5 * self.kn * self.rs
        # 令 x = VGS - Vth  =>  a*x^2 + x + (Vth - VG) = 0
        b, c = 1.0, self.vth - vg
        disc = b * b - 4 * a * c
        if disc < 0:
            return {"region": "截止区", "ID": 0.0, "VGS": vg, "VDS": self.vdd}
        x = (-b + math.sqrt(disc)) / (2 * a)
        if x <= 0:
            return {"region": "截止区 (VGS < Vth)", "ID": 0.0,
                    "VGS": vg, "VDS": self.vdd, "VG": vg}
        vgs = x + self.vth
        idc = 0.5 * self.kn * x * x
        vds = self.vdd - idc * (self.rd + self.rs)
        vov = x                                   # 过驱动电压 Vov = VGS - Vth
        region = "饱和区(放大)" if vds >= vov else "线性区(可变电阻区) —— 需减小 Rd!"
        gm = self.kn * vov                        # gm = kn*Vov = 2*ID/Vov
        ro = 1.0 / (self.lam * idc) if idc > 0 else float("inf")
        return {"VG": vg, "VGS": vgs, "Vov": vov, "ID": idc, "VDS": vds,
                "region": region, "gm": gm, "ro": ro}

    def gain(self):
        q = self.dc_operating_point()
        if q["ID"] <= 0:
            return None
        rd_eff = self.rd if self.rl is None else self.rd * self.rl / (self.rd + self.rl)
        rd_eff = rd_eff * q["ro"] / (rd_eff + q["ro"])
        av = -q["gm"] * rd_eff
        rin = self.r1 * self.r2 / (self.r1 + self.r2)
        return {"Av": av, "Rin": rin, "Rout": self.rd, "Q": q}


def demo_mosfet():
    print("=" * 70)
    print("【4】MOSFET 共源放大器")
    print("=" * 70)
    m = MOSFETAmplifier(vdd=12.0, r1=2e6, r2=1e6, rd=4.7e3, rs=1e3,
                        kn=2e-3, vth=1.5, rl=47e3)
    g = m.gain()
    q = g["Q"]
    print(f"器件参数: kn={m.kn*1e3:.1f} mA/V^2, Vth={m.vth} V, lambda={m.lam}")
    print("-" * 70)
    print(f"  栅极电位   : VG   = {q['VG']:.4f} V")
    print(f"  栅源电压   : VGS  = {q['VGS']:.4f} V   (过驱动 Vov = {q['Vov']:.4f} V)")
    print(f"  漏极电流   : ID   = {q['ID']*1e3:.4f} mA")
    print(f"  漏源电压   : VDS  = {q['VDS']:.4f} V")
    print(f"  工作区     : {q['region']}")
    print(f"  跨导       : gm   = {q['gm']*1e3:.4f} mS   (= 2*ID/Vov)")
    print(f"  输出电阻   : ro   = {q['ro']/1e3:.1f} kOhm")
    print("-" * 70)
    print(f"  电压增益   : Av   = {g['Av']:.3f}  ({20*math.log10(abs(g['Av'])):.2f} dB)")
    print(f"  输入电阻   : Rin  = {g['Rin']/1e6:.2f} MOhm  (远大于 BJT!)")
    print()

    print("--- MOS 与 BJT 跨导对比（同样 1 mA 工作电流）---")
    ic = 1e-3
    gm_bjt = ic / VT
    print(f"  BJT : gm = IC/VT      = {gm_bjt*1e3:.2f} mS")
    for kn in [0.5e-3, 2e-3, 10e-3]:
        vov = math.sqrt(2 * ic / kn)
        gm_mos = kn * vov
        print(f"  MOS : gm = sqrt(2*kn*ID) = {gm_mos*1e3:6.2f} mS  "
              f"(kn={kn*1e3:.1f} mA/V^2, Vov={vov:.3f} V)")
    print("-" * 70)
    print("结论：同电流下 BJT 跨导远高于 MOS —— 这是模拟前端仍偏爱 BJT 的原因；")
    print("      而 MOS 栅极不取电流、易集成、功耗低，是大规模集成的首选。")
    print()


def main():
    demo_bjt_q_point()
    demo_stability()
    demo_configurations()
    demo_mosfet()
    print("=" * 70)
    print("核心方法论：先直流定 Q 点，再交流算增益。两者通过 gm = IC/VT 关联。")
    print("=" * 70)


if __name__ == "__main__":
    main()
