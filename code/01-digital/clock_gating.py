# clock_gating.py —— 时钟门控波形验证 + 功耗账本（仅标准库）
# 对应讲义《数字电子技术》扩展三：低功耗设计与时钟门控
import random

SAMPLES_PER_CYCLE = 4          # 每个时钟周期采 4 点：低,低,高,高


def make_clock(cycles):
    return [0, 0, 1, 1] * cycles


def gate_clock(clk_seq, en_seq, use_latch):
    """两种门控实现。
    use_latch=False：直接 gclk = clk & en（错误做法，en 在时钟高电平期间
                     变化会削出窄脉冲或毛刺）
    use_latch=True ：ICG 单元，先用低电平透明锁存器把 en 稳到时钟低电平期间，
                     再与时钟相与，保证输出永远是完整脉冲
    """
    gclk, latched = [], 0
    for clk, en in zip(clk_seq, en_seq):
        if use_latch and clk == 1:
            pass                      # 时钟高电平时锁存器关闭，保持原值
        else:
            latched = en
        gclk.append(clk & (latched if use_latch else en))
    return gclk


def ascii_wave(name, seq):
    body = "".join("‾" if v else "_" for v in seq)
    return f"{name:>6} |{body}|"


def count_edges(seq):
    return sum(1 for a, b in zip(seq, seq[1:]) if a == 0 and b == 1)


def dynamic_power(c_load, vdd, freq, activity):
    """动态功耗 P = alpha * C * V^2 * f （瓦）"""
    return activity * c_load * vdd ** 2 * freq


if __name__ == "__main__":
    CYCLES = 7
    clk = make_clock(CYCLES)
    # en 在第 2 个周期的高电平中间抬起，第 5 个周期的高电平中间落下
    en = [0] * 11 + [1] * 9 + [0] * (len(clk) - 20)

    print("时钟门控波形对比（‾ = 高，_ = 低，每周期 4 个采样点）")
    print(ascii_wave("CLK", clk))
    print(ascii_wave("EN", en))
    naive = gate_clock(clk, en, use_latch=False)
    icg = gate_clock(clk, en, use_latch=True)
    print(ascii_wave("裸与门", naive))
    print(ascii_wave("ICG", icg))
    print(f"裸与门产生 {count_edges(naive)} 个上升沿，ICG 产生 {count_edges(icg)} 个；")
    print("裸与门在 EN 抬起处削出了一个半宽窄脉冲，会让下游触发器误采样。\n")

    # ---------- 功耗账本 ----------
    random.seed(7)
    N_REG, WIDTH, FREQ = 64, 32, 1e9
    C_FF, C_CLK_LEAF, VDD = 8e-15, 4e-15, 0.9
    write_prob = 0.15                              # 只有 15% 的周期真的要写
    active = [1 if random.random() < write_prob else 0 for _ in range(100000)]
    duty = sum(active) / len(active)

    p_clk_free = dynamic_power(N_REG * WIDTH * C_CLK_LEAF, VDD, FREQ, 1.0)
    p_clk_gated = p_clk_free * duty
    p_data = dynamic_power(N_REG * WIDTH * C_FF, VDD, FREQ, duty * 0.5)

    print(f"64 x 32bit 寄存器堆 @ {FREQ/1e9:.0f}GHz / {VDD}V，实际写活动率 {duty:.1%}")
    print("方案                | 时钟网络功耗 | 数据翻转功耗 | 合计")
    for name, pc, pd, scale in (
        ("无门控",           p_clk_free,  p_data, 1.0),
        ("时钟门控",         p_clk_gated, p_data, 1.0),
        ("门控 + DVFS 0.7V/500MHz",
                             p_clk_gated, p_data, (0.7 / VDD) ** 2 * 0.5),
    ):
        tot = (pc + pd) * scale
        print(f"{name:<20}| {pc*scale*1e3:>9.3f} mW | {pd*scale*1e3:>9.3f} mW "
              f"| {tot*1e3:>7.3f} mW")
    base = p_clk_free + p_data
    print(f"\n只加时钟门控就省掉 {(1-(p_clk_gated+p_data)/base):.1%} 的动态功耗，")
    print("再叠加 DVFS（电压平方项 + 频率线性项）总共可省 90% 以上。")
