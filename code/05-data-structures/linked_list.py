"""
链表与 LRU 缓存 —— 数据结构讲义配套示例
运行：python linked_list.py

包含：
1) 单链表：头插、尾插、按值删除、反转、遍历
2) 招牌案例：LRU 缓存（哈希表 + 双向链表，get/put 均摊 O(1)）
"""


# ============================================================
# 一、单链表
# ============================================================
class Node:
    """单链表节点：数据域 val + 指针域 next"""
    def __init__(self, val):
        self.val = val
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None
        self.size = 0

    def push_front(self, val):
        """头插：O(1)"""
        node = Node(val)
        node.next = self.head
        self.head = node
        self.size += 1

    def push_back(self, val):
        """尾插：O(n)（无尾指针时需遍历到末尾）"""
        node = Node(val)
        if self.head is None:
            self.head = node
        else:
            cur = self.head
            while cur.next:
                cur = cur.next
            cur.next = node
        self.size += 1

    def remove(self, val):
        """按值删除第一个匹配节点：O(n)"""
        dummy = Node(None)      # 哑结点，统一头结点与中间结点的删除逻辑
        dummy.next = self.head
        prev, cur = dummy, self.head
        while cur:
            if cur.val == val:
                prev.next = cur.next
                self.size -= 1
                break
            prev, cur = cur, cur.next
        self.head = dummy.next

    def reverse(self):
        """原地反转：三指针，O(n) 时间 O(1) 空间"""
        prev, cur = None, self.head
        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        self.head = prev

    def to_list(self):
        out, cur = [], self.head
        while cur:
            out.append(cur.val)
            cur = cur.next
        return out


# ============================================================
# 二、招牌案例：LRU 缓存
#   思路：哈希表实现 O(1) 定位，双向链表维护访问顺序
#   头部 = 最近使用，尾部 = 最久未使用（淘汰候选）
# ============================================================
class DNode:
    """双向链表节点，同时保存 key 便于淘汰时从哈希表删除"""
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = {}                 # key -> DNode
        self.head = DNode()             # 哨兵头
        self.tail = DNode()             # 哨兵尾
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_front(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        """访问：命中则移到头部，均摊 O(1)"""
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_front(node)
        return node.val

    def put(self, key, val):
        """写入：已存在则更新并移到头部；超容量则淘汰尾部"""
        if key in self.cache:
            node = self.cache[key]
            node.val = val
            self._remove(node)
            self._add_front(node)
            return
        if len(self.cache) >= self.cap:
            lru = self.tail.prev            # 尾部即最久未使用
            self._remove(lru)
            del self.cache[lru.key]
        node = DNode(key, val)
        self.cache[key] = node
        self._add_front(node)

    def snapshot(self):
        """从头到尾输出当前顺序，用于观察淘汰过程"""
        out, cur = [], self.head.next
        while cur is not self.tail:
            out.append((cur.key, cur.val))
            cur = cur.next
        return out


if __name__ == "__main__":
    print("===== 单链表 =====")
    ll = LinkedList()
    for x in [1, 2, 3]:
        ll.push_back(x)
    ll.push_front(0)
    print("构造后:", ll.to_list())         # [0, 1, 2, 3]
    ll.remove(2)
    print("删除 2 后:", ll.to_list())      # [0, 1, 3]
    ll.reverse()
    print("反转后:", ll.to_list())         # [3, 1, 0]

    print("\n===== LRU 缓存（容量=2）=====")
    lru = LRUCache(2)
    lru.put(1, 100); print("put(1,100):", lru.snapshot())
    lru.put(2, 200); print("put(2,200):", lru.snapshot())
    print("get(1):", lru.get(1), "->", lru.snapshot())   # 1 变为最近使用
    lru.put(3, 300); print("put(3,300) 淘汰最久未用的 2:", lru.snapshot())
    print("get(2):", lru.get(2), "（已被淘汰返回 -1）")
    lru.put(4, 400); print("put(4,400) 淘汰 1:", lru.snapshot())
    print("get(3):", lru.get(3), " get(4):", lru.get(4))
