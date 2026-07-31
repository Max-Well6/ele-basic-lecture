# logic_sim.py —— 真值表生成 + Quine-McCluskey 化简简版（仅标准库）
# 运行：python logic_sim.py
from itertools import product


def truth_table(func, n):
    """打印 n 变量逻辑函数的真值表，返回最小项编号列表"""
    names = [chr(ord('A') + i) for i in range(n)]
    print(" ".join(names) + " | Y")
    minterms = []
    for idx, bits in enumerate(product((0, 1), repeat=n)):
        y = func(*bits)
        print(" ".join(map(str, bits)) + f" | {int(y)}")
        if y:
            minterms.append(idx)
    return minterms


def qm_simplify(minterms, n):
    """Quine-McCluskey 简版：返回素蕴含项的字符串形式（'-'表示消去的变量）"""
    terms = {format(m, f"0{n}b") for m in minterms}
    primes = set()
    while terms:
        merged, used = set(), set()
        term_list = sorted(terms)
        for i in range(len(term_list)):
            for j in range(i + 1, len(term_list)):
                a, b = term_list[i], term_list[j]
                diff = [k for k in range(n) if a[k] != b[k]]
                if len(diff) == 1 and a[diff[0]] != '-' and b[diff[0]] != '-':
                    merged.add(a[:diff[0]] + '-' + a[diff[0]+1:])
                    used.update((a, b))
        primes |= terms - used
        terms = merged
    return sorted(primes)


def to_expr(implicant):
    """把 '1-0' 这类蕴含项转成 A·C' 形式"""
    out = []
    for i, c in enumerate(implicant):
        v = chr(ord('A') + i)
        if c == '1':
            out.append(v)
        elif c == '0':
            out.append(v + "'")
    return "·".join(out) if out else "1"


if __name__ == "__main__":
    # 示例：三变量多数表决函数 Y = AB + BC + AC
    f = lambda a, b, c: (a & b) | (b & c) | (a & c)
    ms = truth_table(f, 3)
    print("最小项:", ms)                       # [3, 5, 6, 7]
    primes = qm_simplify(ms, 3)
    print("化简结果: Y = " + " + ".join(to_expr(p) for p in primes))
