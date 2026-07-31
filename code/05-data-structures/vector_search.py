"""
迷你向量检索引擎 —— 数据结构讲义配套示例（呼应 AI 时代）
运行：python vector_search.py

只用标准库实现一个"能跑的最小 RAG 检索层"：
1) 文本 -> 向量：字符 bigram 词袋 + L2 归一化（简化版嵌入）
2) 相似度：余弦相似度
3) TopK：暴力扫描 + 大小为 K 的小顶堆（O(n log k) 而非全排序 O(n log n)）
4) 倒排索引加速：只扫描含公共特征的候选集，体现"哈希表换时间"
5) 布隆过滤器：O(1) 判断"这段文本是否已入库"，省一次全量查表
"""

import heapq
import math
import hashlib
from collections import defaultdict


# ============================================================
# 一、极简"嵌入"：字符 bigram 词袋 + L2 归一化
#    真实系统用神经网络产出稠密向量，这里用稀疏字典模拟，原理相同
# ============================================================
def embed(text):
    """返回 {特征: 权重} 的稀疏向量，已 L2 归一化"""
    text = text.strip().lower()
    vec = defaultdict(float)
    for ch in text:                                 # unigram
        if not ch.isspace():
            vec[ch] += 1.0
    for i in range(len(text) - 1):                  # bigram
        gram = text[i:i + 2]
        if not gram.isspace():
            vec[gram] += 1.5                        # 二元组信息量更大，加权
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def cosine(a, b):
    """两个稀疏向量的余弦相似度；已归一化时点积即余弦"""
    if len(a) > len(b):
        a, b = b, a                                 # 遍历较短的那个
    return sum(w * b.get(k, 0.0) for k, w in a.items())


# ============================================================
# 二、布隆过滤器：位数组 + k 个哈希函数
#    特点：可能误判"存在"，绝不误判"不存在"
# ============================================================
class BloomFilter:
    def __init__(self, m_bits=1 << 14, k=3):
        self.m = m_bits
        self.k = k
        self.bits = bytearray(m_bits // 8)

    def _hashes(self, s):
        digest = hashlib.sha256(s.encode("utf-8")).digest()
        # 从摘要里切出 k 个独立哈希值
        for i in range(self.k):
            chunk = digest[i * 4:(i + 1) * 4]
            yield int.from_bytes(chunk, "big") % self.m

    def add(self, s):
        for h in self._hashes(s):
            self.bits[h // 8] |= 1 << (h % 8)

    def __contains__(self, s):
        return all(self.bits[h // 8] & (1 << (h % 8)) for h in self._hashes(s))


# ============================================================
# 三、向量库：暴力检索 + 堆 TopK + 倒排索引加速
# ============================================================
class VectorStore:
    def __init__(self):
        self.docs = []                              # [(id, 文本, 向量)]
        self.inverted = defaultdict(list)           # 特征 -> [doc_id]
        self.bloom = BloomFilter()

    def add(self, text):
        """入库；用布隆过滤器先做一次 O(1) 去重预判"""
        if text in self.bloom:
            # 可能误判，需回查确认（布隆只用于剪枝）
            if any(t == text for _, t, _ in self.docs):
                print(f"  [跳过重复] {text}")
                return
        vec = embed(text)
        doc_id = len(self.docs)
        self.docs.append((doc_id, text, vec))
        for feat in vec:
            self.inverted[feat].append(doc_id)
        self.bloom.add(text)

    def search_bruteforce(self, query, k=3):
        """暴力：算全部 n 个相似度，再用小顶堆维护 TopK
           时间 O(n·d + n log k)，空间 O(k)"""
        qv = embed(query)
        heap = []                                   # 小顶堆，堆顶是当前第 k 名
        for doc_id, text, vec in self.docs:
            score = cosine(qv, vec)
            if len(heap) < k:
                heapq.heappush(heap, (score, doc_id, text))
            elif score > heap[0][0]:                # 比第 k 名好才替换
                heapq.heapreplace(heap, (score, doc_id, text))
        return sorted(heap, key=lambda x: -x[0])

    def search_inverted(self, query, k=3):
        """倒排加速：只对与 query 有公共特征的候选文档打分"""
        qv = embed(query)
        candidates = set()
        for feat in qv:
            candidates.update(self.inverted.get(feat, ()))
        heap = []
        for doc_id in candidates:
            _, text, vec = self.docs[doc_id]
            score = cosine(qv, vec)
            if len(heap) < k:
                heapq.heappush(heap, (score, doc_id, text))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, doc_id, text))
        return sorted(heap, key=lambda x: -x[0]), len(candidates)


if __name__ == "__main__":
    corpus = [
        "二叉搜索树的中序遍历结果是有序序列",
        "哈希表通过哈希函数把键映射到数组下标",
        "堆是一棵完全二叉树，常用于实现优先队列",
        "红黑树是一种自平衡的二叉搜索树",
        "布隆过滤器用位数组判断元素是否可能存在",
        "图的广度优先搜索使用队列逐层扩散",
        "B+树是数据库索引的主流数据结构",
        "向量数据库用近似最近邻算法检索嵌入向量",
        "栈是后进先出的线性表，函数调用依赖它",
        "并查集可以高效判断两个元素是否属于同一集合",
    ]

    store = VectorStore()
    print("===== 建库 =====")
    for text in corpus:
        store.add(text)
    store.add(corpus[0])                            # 测试去重
    print(f"入库文档数: {len(store.docs)}，倒排特征数: {len(store.inverted)}")

    queries = ["平衡的二叉树", "数据库索引用什么结构", "先进先出的队列"]

    print("\n===== 暴力 + 堆 TopK 检索 =====")
    for q in queries:
        print(f"\n查询: {q}")
        for rank, (score, doc_id, text) in enumerate(store.search_bruteforce(q, 3), 1):
            print(f"  Top{rank} 相似度={score:.4f}  #{doc_id} {text}")

    print("\n===== 倒排索引加速（结果一致但候选更少）=====")
    q = "数据库索引用什么结构"
    results, n_cand = store.search_inverted(q, 3)
    print(f"查询: {q}（候选 {n_cand}/{len(store.docs)} 篇）")
    for rank, (score, doc_id, text) in enumerate(results, 1):
        print(f"  Top{rank} 相似度={score:.4f}  #{doc_id} {text}")

    print("\n===== 布隆过滤器 =====")
    print("  '栈是后进先出的线性表，函数调用依赖它' in bloom:",
          "栈是后进先出的线性表，函数调用依赖它" in store.bloom)
    print("  '这句话从未入库过' in bloom:", "这句话从未入库过" in store.bloom)
