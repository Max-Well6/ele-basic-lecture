# fsm_1011.py —— 可重叠检测 1011 的 Mealy 状态机（仅标准库）
import random

class Mealy1011:
    """可重叠检测 1011。状态含义 = 已匹配的最长前缀。"""
    S0, S1, S10, S101 = 0, 1, 2, 3
    # (现态, 输入) -> (次态, 输出)
    TABLE = {
        (S0,   0): (S0,   0), (S0,   1): (S1,   0),
        (S1,   0): (S10,  0), (S1,   1): (S1,   0),
        (S10,  0): (S0,   0), (S10,  1): (S101, 0),
        (S101, 0): (S10,  0), (S101, 1): (S1,   1),   # 检出并重叠
    }

    def __init__(self):
        self.state = self.S0

    def reset(self):
        self.state = self.S0

    def step(self, bit: int) -> int:
        self.state, out = self.TABLE[(self.state, bit)]
        return out

def naive_count(bits):
    """朴素字符串匹配作为参考模型（允许重叠）"""
    s = "".join(map(str, bits))
    return sum(1 for i in range(len(s) - 3) if s[i:i + 4] == "1011")

if __name__ == "__main__":
    fsm, seq = Mealy1011(), [1, 0, 1, 1, 0, 1, 1]
    print("输入 1011011 的逐拍输出:", [fsm.step(b) for b in seq])
    print("朴素法检出次数:", naive_count(seq))

    random.seed(42)
    for _ in range(1000):                       # 随机对拍 1000 次
        bits = [random.randint(0, 1) for _ in range(random.randint(4, 40))]
        fsm = Mealy1011()
        assert sum(fsm.step(b) for b in bits) == naive_count(bits), bits
    print("PASS: 1000 组随机序列与朴素匹配结果完全一致")
