# fsm_vending.py —— 自动售货机 Moore 状态机模拟（仅标准库）
# 商品 15 元，接受投币 5 元 / 10 元，支持找零
# 运行：python fsm_vending.py


class VendingMachine:
    """状态 = 已投金额（0/5/10），Moore 输出由状态决定"""
    STATES = (0, 5, 10)

    def __init__(self):
        self.state = 0

    def insert(self, coin: int):
        """投入一枚硬币，返回 (是否出货, 找零金额)"""
        if coin not in (5, 10):
            raise ValueError("只接受 5 元或 10 元硬币")
        total = self.state + coin
        if total >= 15:
            change = total - 15
            self.state = 0                 # 出货后回到初始状态
            return True, change
        self.state = total
        return False, 0


if __name__ == "__main__":
    vm = VendingMachine()
    coin_seq = [5, 5, 5, 10, 10, 5, 10]
    for coin in coin_seq:
        prev = vm.state
        dispense, change = vm.insert(coin)
        print(f"状态 {prev:>2} --投{coin}元--> 状态 {vm.state:>2} "
              f"| 出货={'是' if dispense else '否'} 找零={change}")
