"""
二叉搜索树与堆 —— 数据结构讲义配套示例
运行：python bst.py

包含：
1) BST：插入（打印每步中序遍历）、查找、删除、前中后序 + 层序遍历、树高
2) AVL 平衡树：四种旋转（LL/RR/LR/RL），打印每次插入的失衡与旋转过程
3) 二叉堆：手写小顶堆（sift_up / sift_down），打印建堆的逐步调整过程
4) 堆排序
"""


# ============================================================
# 一、二叉搜索树 BST
#   性质：左子树所有键 < 根 < 右子树所有键 => 中序遍历必然有序
# ============================================================
class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        """插入：平均 O(log n)，最坏（退化成链）O(n)"""
        def _insert(node, key):
            if node is None:
                return TreeNode(key)
            if key < node.key:
                node.left = _insert(node.left, key)
            elif key > node.key:
                node.right = _insert(node.right, key)
            # key 相等时忽略（不存重复键）
            return node
        self.root = _insert(self.root, key)

    def search(self, key):
        """查找：从根开始每次砍掉一半，平均 O(log n)"""
        cur, path = self.root, []
        while cur:
            path.append(cur.key)
            if key == cur.key:
                return True, path
            cur = cur.left if key < cur.key else cur.right
        return False, path

    def delete(self, key):
        """删除三种情况：叶子 / 单子树 / 双子树（用右子树最小值顶替）"""
        def _min_node(node):
            while node.left:
                node = node.left
            return node

        def _delete(node, key):
            if node is None:
                return None
            if key < node.key:
                node.left = _delete(node.left, key)
            elif key > node.key:
                node.right = _delete(node.right, key)
            else:
                if node.left is None:
                    return node.right
                if node.right is None:
                    return node.left
                succ = _min_node(node.right)    # 后继：右子树最小
                node.key = succ.key
                node.right = _delete(node.right, succ.key)
            return node
        self.root = _delete(self.root, key)

    def inorder(self):
        """中序：左-根-右，结果升序"""
        out = []
        def _go(n):
            if n:
                _go(n.left); out.append(n.key); _go(n.right)
        _go(self.root)
        return out

    def preorder(self):
        out = []
        def _go(n):
            if n:
                out.append(n.key); _go(n.left); _go(n.right)
        _go(self.root)
        return out

    def postorder(self):
        out = []
        def _go(n):
            if n:
                _go(n.left); _go(n.right); out.append(n.key)
        _go(self.root)
        return out

    def levelorder(self):
        """层序：借助队列 BFS"""
        if not self.root:
            return []
        out, queue = [], [self.root]
        while queue:
            node = queue.pop(0)
            out.append(node.key)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return out

    def height(self):
        def _h(n):
            return 0 if n is None else 1 + max(_h(n.left), _h(n.right))
        return _h(self.root)


# ============================================================
# 二、AVL 平衡二叉搜索树
#   平衡条件：任一节点左右子树高度差不超过 1
#   失衡后靠旋转恢复，保证树高始终是 O(log n)
# ============================================================
class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = self.right = None
        self.height = 1


def h(node):
    return node.height if node else 0


def _update(node):
    node.height = 1 + max(h(node.left), h(node.right))


def balance_factor(node):
    """平衡因子 = 左子树高 - 右子树高，绝对值 > 1 即失衡"""
    return h(node.left) - h(node.right) if node else 0


def rotate_right(y):
    r"""右旋，处理 LL 型失衡（中序序列保持不变）
         y              x
        / \            / \
       x   C   ==>    A   y
      / \                / \
     A   B              B   C
    """
    x = y.left
    y.left, x.right = x.right, y
    _update(y)
    _update(x)
    return x


def rotate_left(x):
    """左旋，处理 RR 型失衡（右旋的镜像）"""
    y = x.right
    x.right, y.left = y.left, x
    _update(x)
    _update(y)
    return y


def avl_insert(node, key, log):
    """插入并在回溯途中检查平衡，log 记录发生的旋转"""
    if node is None:
        return AVLNode(key)
    if key < node.key:
        node.left = avl_insert(node.left, key, log)
    elif key > node.key:
        node.right = avl_insert(node.right, key, log)
    else:
        return node
    _update(node)

    bf = balance_factor(node)
    if bf > 1 and key < node.left.key:              # LL
        log.append(f"节点 {node.key} LL 失衡 -> 右旋")
        return rotate_right(node)
    if bf < -1 and key > node.right.key:            # RR
        log.append(f"节点 {node.key} RR 失衡 -> 左旋")
        return rotate_left(node)
    if bf > 1 and key > node.left.key:              # LR
        log.append(f"节点 {node.key} LR 失衡 -> 左孩子左旋 + 自身右旋")
        node.left = rotate_left(node.left)
        return rotate_right(node)
    if bf < -1 and key < node.right.key:            # RL
        log.append(f"节点 {node.key} RL 失衡 -> 右孩子右旋 + 自身左旋")
        node.right = rotate_right(node.right)
        return rotate_left(node)
    return node


def plain_bst_insert(root, key):
    """不做平衡的普通 BST，用于对照退化情况"""
    if root is None:
        return AVLNode(key)
    if key < root.key:
        root.left = plain_bst_insert(root.left, key)
    else:
        root.right = plain_bst_insert(root.right, key)
    _update(root)
    return root


def level_view(root):
    """按层输出各层的键，直观展示树形"""
    levels, cur = [], [root] if root else []
    while cur:
        levels.append([n.key for n in cur])
        cur = [c for n in cur for c in (n.left, n.right) if c]
    return levels


# ============================================================
# 三、二叉小顶堆（数组表示：下标 i 的左右孩子为 2i+1 / 2i+2）
# ============================================================
class MinHeap:
    def __init__(self, data=None, verbose=False):
        self.a = list(data) if data else []
        self.verbose = verbose
        if self.a:
            self._build()

    def _build(self):
        """自底向上建堆：从最后一个非叶子节点开始下沉，O(n)"""
        n = len(self.a)
        for i in range(n // 2 - 1, -1, -1):
            if self.verbose:
                print(f"  下沉下标 {i}(值={self.a[i]}) 前: {self.a}")
            self._sift_down(i)
            if self.verbose:
                print(f"  下沉下标 {i} 后: {self.a}")

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.a[i] < self.a[parent]:
                self.a[i], self.a[parent] = self.a[parent], self.a[i]
                i = parent
            else:
                break

    def _sift_down(self, i, n=None):
        n = len(self.a) if n is None else n
        while True:
            smallest, l, r = i, 2 * i + 1, 2 * i + 2
            if l < n and self.a[l] < self.a[smallest]:
                smallest = l
            if r < n and self.a[r] < self.a[smallest]:
                smallest = r
            if smallest == i:
                break
            self.a[i], self.a[smallest] = self.a[smallest], self.a[i]
            i = smallest

    def push(self, x):
        """插入：放末尾后上浮，O(log n)"""
        self.a.append(x)
        self._sift_up(len(self.a) - 1)

    def pop(self):
        """弹出最小值：堆顶与末尾交换后下沉，O(log n)"""
        if not self.a:
            raise IndexError("heap is empty")
        top = self.a[0]
        last = self.a.pop()
        if self.a:
            self.a[0] = last
            self._sift_down(0)
        return top

    def peek(self):
        return self.a[0]

    def __len__(self):
        return len(self.a)


def heap_sort(arr):
    """堆排序：先建大顶堆，再反复把堆顶换到末尾，O(n log n) 原地排序"""
    a = list(arr)
    n = len(a)

    def sift_down_max(i, size):
        while True:
            largest, l, r = i, 2 * i + 1, 2 * i + 2
            if l < size and a[l] > a[largest]:
                largest = l
            if r < size and a[r] > a[largest]:
                largest = r
            if largest == i:
                break
            a[i], a[largest] = a[largest], a[i]
            i = largest

    for i in range(n // 2 - 1, -1, -1):
        sift_down_max(i, n)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down_max(0, end)
    return a


if __name__ == "__main__":
    print("===== 二叉搜索树 =====")
    bst = BST()
    for k in [50, 30, 70, 20, 40, 60, 80]:
        bst.insert(k)
        print(f"插入 {k:>2} 后中序: {bst.inorder()}")

    print("\n前序:", bst.preorder())
    print("后序:", bst.postorder())
    print("层序:", bst.levelorder())
    print("树高:", bst.height())

    found, path = bst.search(60)
    print(f"查找 60 -> {found}, 比较路径: {path}")
    found, path = bst.search(45)
    print(f"查找 45 -> {found}, 比较路径: {path}")

    bst.delete(20); print("删除叶子 20:", bst.inorder())
    bst.delete(30); print("删除单子树 30:", bst.inorder())
    bst.delete(50); print("删除双子树根 50:", bst.inorder())

    print("\n===== AVL 平衡树：依次插入 10,20,30,40,50,25 =====")
    avl_root, plain_root, log = None, None, []
    for k in [10, 20, 30, 40, 50, 25]:
        log.clear()
        avl_root = avl_insert(avl_root, k, log)
        plain_root = plain_bst_insert(plain_root, k)
        print(f"插入 {k:>2}: {'；'.join(log) if log else '无需旋转'}")
        print(f"        层序 {level_view(avl_root)}  树高={h(avl_root)}")
    print("对照 普通 BST 层序:", level_view(plain_root),
          " 树高 =", h(plain_root), "（明显退化）")

    print("\n===== 小顶堆建堆过程 =====")
    h = MinHeap([9, 4, 7, 1, 8, 3, 6], verbose=True)
    print("建堆结果:", h.a)
    h.push(0)
    print("push(0) 后:", h.a)
    order = []
    while len(h):
        order.append(h.pop())
    print("依次 pop 得到升序:", order)

    print("\n===== 堆排序 =====")
    data = [5, 2, 9, 1, 7, 3, 8, 6]
    print("原始:", data)
    print("排序:", heap_sort(data))
