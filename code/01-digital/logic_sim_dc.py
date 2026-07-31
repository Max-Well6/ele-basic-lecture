# logic_sim_dc.py —— 带无关项的 Quine-McCluskey 化简（仅标准库）

def qm_simplify(minterms, dontcares, n):
    """无关项参与合并，但不要求被覆盖。返回选中的素蕴含项列表。"""
    # ---- 第 1 步：无关项一起参与合并，求全部素蕴含项 ----
    terms = {format(m, f"0{n}b") for m in set(minterms) | set(dontcares)}
    primes = set()
    while terms:
        merged, used = set(), set()
        tl = sorted(terms)
        for i in range(len(tl)):
            for j in range(i + 1, len(tl)):
                a, b = tl[i], tl[j]
                diff = [k for k in range(n) if a[k] != b[k]]
                if len(diff) == 1 and a[diff[0]] != '-' and b[diff[0]] != '-':
                    merged.add(a[:diff[0]] + '-' + a[diff[0] + 1:])
                    used.update((a, b))
        primes |= terms - used          # 没能再合并的就是素蕴含项
        terms = merged

    # ---- 第 2 步：覆盖表只放真最小项，无关项不参与覆盖 ----
    def covers(p, m):
        s = format(m, f"0{n}b")
        return all(c == '-' or c == s[k] for k, c in enumerate(p))

    need = sorted(set(minterms))
    chosen = set()
    for m in need:                       # 必要素蕴含项：只有唯一选择的
        ps = [p for p in primes if covers(p, m)]
        if len(ps) == 1:
            chosen.add(ps[0])
    rest = [m for m in need if not any(covers(p, m) for p in chosen)]
    while rest:                          # 贪心补齐：每次选覆盖最多的
        best = max(primes - chosen,
                   key=lambda p: (sum(covers(p, m) for m in rest),
                                  p.count('-')))
        chosen.add(best)
        rest = [m for m in rest if not covers(best, m)]
    return sorted(chosen)

def to_expr(imp):
    out = []
    for i, c in enumerate(imp):
        v = chr(ord('A') + i)
        if c == '1':
            out.append(v)
        elif c == '0':
            out.append(v + "'")
    return "·".join(out) if out else "1"

if __name__ == "__main__":
    # BCD(A B C D，A 为最高位) 转格雷码，10~15 是无关项
    dc = list(range(10, 16))
    for bit in range(4):                 # 分别化简 G3..G0 四个输出
        ms = [b for b in range(10) if ((b ^ (b >> 1)) >> bit) & 1]
        prim = qm_simplify(ms, dc, 4)
        print(f"G{bit}: 最小项={ms}")
        print(f"     化简 = " + " + ".join(to_expr(p) for p in prim))
        for v in range(10):              # 自检：与 B^(B>>1) 逐个比对
            s = format(v, "04b")
            got = int(any(all(c == '-' or c == s[k]
                              for k, c in enumerate(p)) for p in prim))
            assert got == ((v ^ (v >> 1)) >> bit) & 1
    print("PASS: 4 位输出在 0~9 上全部与 B^(B>>1) 一致")
