"""
排序算法全景与性能实测
======================
包含 6 种经典排序算法的教学实现，并在随机数据上做计时对比。

运行方式：
    python sorting.py

教学要点：
1. 比较类排序的理论下界是 O(n log n)（决策树高度 log(n!) = Theta(n log n)）；
2. 计数排序等非比较类排序可以做到 O(n + k)，但依赖数据范围 k；
3. 稳定性（stable）指相等元素的相对次序在排序后保持不变；
4. 大 O 相同不等于实际速度相同——常数因子、缓存友好性、语言实现都影响真实耗时。
"""

import random
import time


# ----------------------------------------------------------------------
# 1. 冒泡排序 O(n^2)，稳定
# ----------------------------------------------------------------------
def bubble_sort(a):
    """每一轮把当前最大元素"冒泡"到末尾；加入 swapped 标志以适应近似有序数据。"""
    a = a[:]  # 不修改调用者的列表
    n = len(a)
    for i in range(n - 1):
        swapped = False
        # 末尾 i 个元素已经就位，无需再比较
        for j in range(n - 1 - i):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:      # 一轮下来没有交换，说明已经有序
            break
    return a


# ----------------------------------------------------------------------
# 2. 插入排序 O(n^2)，稳定；小规模数据的实战冠军
# ----------------------------------------------------------------------
def insertion_sort(a):
    """把 a[i] 插入到左侧已排好序的区间 a[0..i-1] 中的正确位置。"""
    a = a[:]
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        # 用 > 而不是 >=，保证相等元素不发生交换 —— 这正是"稳定"的来源
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


# ----------------------------------------------------------------------
# 3. 归并排序 O(n log n)，稳定，需要 O(n) 额外空间
# ----------------------------------------------------------------------
def merge_sort(a):
    """分治三步：分（对半切）→ 治（递归排序）→ 合（线性归并）。
    递归式：T(n) = 2T(n/2) + O(n)  =>  T(n) = O(n log n)
    """
    if len(a) <= 1:
        return a[:]
    mid = len(a) // 2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])
    return _merge(left, right)


def _merge(left, right):
    """合并两个有序表；<= 保证左半部分优先，因此归并排序是稳定的。"""
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


# ----------------------------------------------------------------------
# 4. 快速排序 平均 O(n log n)，最坏 O(n^2)，不稳定
# ----------------------------------------------------------------------
def quick_sort(a):
    """三路切分（Dutch National Flag）版本，能优雅处理大量重复元素。
    随机选主元可以把"最坏情况"变成小概率事件。
    """
    a = a[:]
    _quick_sort_inplace(a, 0, len(a) - 1)
    return a


def _quick_sort_inplace(a, lo, hi):
    # 小区间改用插入排序，是工业实现的常见优化；这里为了教学保持简单
    if lo >= hi:
        return
    pivot = a[random.randint(lo, hi)]     # 随机主元，避免有序输入退化
    lt, i, gt = lo, lo, hi                # 不变式：[lo,lt) < pivot, [lt,i) == pivot, (gt,hi] > pivot
    while i <= gt:
        if a[i] < pivot:
            a[lt], a[i] = a[i], a[lt]
            lt += 1
            i += 1
        elif a[i] > pivot:
            a[i], a[gt] = a[gt], a[i]
            gt -= 1                        # 换过来的元素还没检查，i 不前进
        else:
            i += 1
    _quick_sort_inplace(a, lo, lt - 1)
    _quick_sort_inplace(a, gt + 1, hi)


# ----------------------------------------------------------------------
# 5. 堆排序 O(n log n)，原地，不稳定
# ----------------------------------------------------------------------
def heap_sort(a):
    """先自底向上建大顶堆 O(n)，再反复把堆顶换到末尾并下沉修复 O(n log n)。"""
    a = a[:]
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):    # 从最后一个非叶子结点开始建堆
        _sift_down(a, i, n)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]        # 最大值归位
        _sift_down(a, 0, end)              # 在缩小的堆上修复
    return a


def _sift_down(a, root, size):
    while True:
        child = 2 * root + 1
        if child >= size:
            break
        if child + 1 < size and a[child + 1] > a[child]:
            child += 1                     # 选较大的孩子
        if a[root] >= a[child]:
            break
        a[root], a[child] = a[child], a[root]
        root = child


# ----------------------------------------------------------------------
# 6. 计数排序 O(n + k)，稳定，非比较类，只适用于小范围整数
# ----------------------------------------------------------------------
def counting_sort(a):
    """统计每个值出现的次数，再按前缀和把元素放回原位置。
    k = max - min + 1；当 k 远大于 n 时（如排序 32 位随机整数）会浪费大量内存。
    """
    if not a:
        return []
    lo, hi = min(a), max(a)
    k = hi - lo + 1
    count = [0] * k
    for x in a:
        count[x - lo] += 1
    for i in range(1, k):
        count[i] += count[i - 1]           # 前缀和 = 每个值的结束下标
    out = [0] * len(a)
    for x in reversed(a):                  # 倒序遍历保证稳定性
        count[x - lo] -= 1
        out[count[x - lo]] = x
    return out


# ----------------------------------------------------------------------
# 性能实测
# ----------------------------------------------------------------------
def benchmark(sizes=(1000, 4000, 16000)):
    """在随机数据上对比各算法耗时，输出 Markdown 风格表格。"""
    algorithms = [
        ("冒泡排序", bubble_sort, 4000),      # 第三列：可承受的最大规模，超过就跳过
        ("插入排序", insertion_sort, 4000),
        ("归并排序", merge_sort, 10 ** 9),
        ("快速排序", quick_sort, 10 ** 9),
        ("堆排序", heap_sort, 10 ** 9),
        ("计数排序", counting_sort, 10 ** 9),
        ("内置 sorted", sorted, 10 ** 9),
    ]

    header = "| 算法 | " + " | ".join(f"n={n}" for n in sizes) + " |"
    sep = "|------|" + "|".join(["--------"] * len(sizes)) + "|"
    print(header)
    print(sep)

    for name, func, limit in algorithms:
        cells = []
        for n in sizes:
            if n > limit:
                cells.append("  跳过  ")
                continue
            data = [random.randint(0, n) for _ in range(n)]
            expected = sorted(data)
            t0 = time.perf_counter()
            result = func(data)
            elapsed = time.perf_counter() - t0
            # 正确性校验：任何一次结果不对都要立刻暴露
            assert result == expected, f"{name} 排序结果错误"
            cells.append(f"{elapsed * 1000:7.2f}ms")
        print(f"| {name} | " + " | ".join(cells) + " |")


def demo_stability():
    """用 (分数, 姓名) 二元组演示稳定性：分数相同的人，原始顺序是否被保留。"""
    records = [(90, "赵"), (85, "钱"), (90, "孙"), (85, "李"), (95, "周")]
    print("原始数据：", records)

    # 只按分数排序。稳定算法应保持 赵 在 孙 前、钱 在 李 前
    def by_score_insertion(rs):
        rs = rs[:]
        for i in range(1, len(rs)):
            key = rs[i]
            j = i - 1
            while j >= 0 and rs[j][0] > key[0]:
                rs[j + 1] = rs[j]
                j -= 1
            rs[j + 1] = key
        return rs

    print("插入排序(稳定)：", by_score_insertion(records))
    print("Python sorted(稳定)：", sorted(records, key=lambda r: r[0]))


if __name__ == "__main__":
    random.seed(42)          # 固定随机种子，保证结果可复现

    print("=" * 60)
    print("【1】正确性自检（小数据）")
    print("=" * 60)
    sample = [5, 2, 9, 2, 7, 1, 8, 2]
    print("输入：", sample)
    for name, func in [("冒泡", bubble_sort), ("插入", insertion_sort),
                       ("归并", merge_sort), ("快排", quick_sort),
                       ("堆排", heap_sort), ("计数", counting_sort)]:
        print(f"  {name}排序 -> {func(sample)}")

    print()
    print("=" * 60)
    print("【2】稳定性演示")
    print("=" * 60)
    demo_stability()

    print()
    print("=" * 60)
    print("【3】性能实测（随机数据，单位：毫秒）")
    print("=" * 60)
    benchmark()
    print()
    print("结论：O(n^2) 算法在 n 增大 4 倍时耗时约增大 16 倍；")
    print("      O(n log n) 算法耗时约增大 4 倍多一点；")
    print("      内置 sorted (Timsort, C 实现) 常数因子最小，工程中优先使用。")
