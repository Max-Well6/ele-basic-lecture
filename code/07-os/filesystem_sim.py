# -*- coding: utf-8 -*-
"""简易 inode 文件系统模拟。

磁盘模型：64 个数据块，每块 16 字节；inode 含直接指针 x4 + 一级间接指针 x1，
故单文件最大 (4 + 16/2) x 16 = 192 字节（间接块每 2 字节存一个块号，可存 8 个指针，
这里简化为一个 Python 列表，容量 8）。演示创建 / 写入 / 读取 / 删除全过程，
展示 inode 与数据块位图的分配与回收。只依赖标准库。
"""

BLOCK_SIZE = 16      # 每块字节数
N_BLOCKS = 64        # 数据块总数
N_INODES = 8         # inode 总数
N_DIRECT = 4         # 直接指针数
N_INDIRECT = 8       # 一级间接块可容纳的指针数


class Inode:
    def __init__(self):
        self.used = False
        self.size = 0
        self.direct = [None] * N_DIRECT   # 直接指针
        self.indirect = None              # 一级间接指针（指向一个"指针块"）


class FileSystem:
    def __init__(self):
        self.blocks = [b""] * N_BLOCKS            # 数据块
        self.block_bitmap = [False] * N_BLOCKS    # 数据块位图
        self.inodes = [Inode() for _ in range(N_INODES)]
        self.root = {}                            # 根目录：文件名 -> inode 号

    # ---------------- 底层分配 ----------------
    def _alloc_block(self):
        for i, used in enumerate(self.block_bitmap):
            if not used:
                self.block_bitmap[i] = True
                return i
        raise OSError("磁盘满：没有空闲数据块")

    def _free_block(self, bno):
        self.block_bitmap[bno] = False
        self.blocks[bno] = b""

    def _data_blocks(self, ino):
        """按顺序返回文件占用的所有数据块号。"""
        node = self.inodes[ino]
        blocks = [b for b in node.direct if b is not None]
        if node.indirect is not None:
            ptrs = self.blocks[node.indirect]     # 间接块里存的是块号列表
            blocks += list(ptrs)
        return blocks

    # ---------------- 文件操作 ----------------
    def create(self, name):
        if name in self.root:
            raise OSError(f"文件已存在: {name}")
        for i, node in enumerate(self.inodes):
            if not node.used:
                node.used = True
                self.root[name] = i
                print(f"create('{name}') -> 分配 inode #{i}")
                return
        raise OSError("inode 用尽")

    def write(self, name, data: bytes):
        ino = self.root[name]
        node = self.inodes[ino]
        self._truncate(node)                       # 简化：覆盖写，先清空旧块
        n_blk = (len(data) + BLOCK_SIZE - 1) // BLOCK_SIZE
        if n_blk > N_DIRECT + N_INDIRECT:
            raise OSError("文件过大")
        got = []
        for k in range(n_blk):
            bno = self._alloc_block()
            self.blocks[bno] = data[k * BLOCK_SIZE:(k + 1) * BLOCK_SIZE]
            got.append(bno)
            if k < N_DIRECT:
                node.direct[k] = bno               # 前 4 块走直接指针
            else:
                if node.indirect is None:          # 超出部分启用间接块
                    node.indirect = self._alloc_block()
                    self.blocks[node.indirect] = []
                    print(f"  文件超过 {N_DIRECT} 块，分配一级间接块 #{node.indirect}")
                self.blocks[node.indirect] = list(self.blocks[node.indirect]) + [bno]
        node.size = len(data)
        print(f"write('{name}', {len(data)}B) -> 占用数据块 {got}"
              f"（直接 {got[:N_DIRECT]}，间接 {got[N_DIRECT:] or '无'}）")

    def read(self, name) -> bytes:
        ino = self.root[name]
        data = b"".join(self.blocks[b] for b in self._data_blocks(ino))
        return data[: self.inodes[ino].size]

    def _truncate(self, node):
        for b in [x for x in node.direct if x is not None]:
            self._free_block(b)
        if node.indirect is not None:
            for b in self.blocks[node.indirect]:
                self._free_block(b)
            self._free_block(node.indirect)
        node.direct = [None] * N_DIRECT
        node.indirect = None
        node.size = 0

    def delete(self, name):
        ino = self.root.pop(name)
        node = self.inodes[ino]
        freed = self._data_blocks(ino)
        self._truncate(node)
        node.used = False
        print(f"delete('{name}') -> 回收 inode #{ino} 与数据块 {freed}")

    # ---------------- 状态展示 ----------------
    def stat(self):
        used_blk = sum(self.block_bitmap)
        used_ino = sum(n.used for n in self.inodes)
        print(f"[磁盘状态] inode: {used_ino}/{N_INODES} 已用，"
              f"数据块: {used_blk}/{N_BLOCKS} 已用，目录: {dict(self.root)}")
        for name, ino in self.root.items():
            node = self.inodes[ino]
            print(f"  '{name}': inode#{ino} size={node.size}B "
                  f"direct={node.direct} indirect={node.indirect}")


if __name__ == "__main__":
    fs = FileSystem()

    print("===== 1. 创建并写入一个小文件（只用直接指针）=====")
    fs.create("hello.txt")
    fs.write("hello.txt", b"Hello, inode file system!")   # 25B -> 2 块
    fs.stat()
    print("读出内容:", fs.read("hello.txt").decode())

    print("\n===== 2. 写入大文件（触发一级间接块）=====")
    fs.create("big.bin")
    big = bytes(range(96))                                # 96B -> 6 块 > 4 直接指针
    fs.write("big.bin", big)
    fs.stat()
    print("读回校验:", "一致" if fs.read("big.bin") == big else "不一致!")

    print("\n===== 3. 覆盖写：旧块回收、新块分配 =====")
    fs.write("hello.txt", b"short")                       # 变小，旧 2 块回收成 1 块
    fs.stat()

    print("\n===== 4. 删除文件：inode 与数据块全部回收 =====")
    fs.delete("big.bin")
    fs.delete("hello.txt")
    fs.stat()
    print("\n结论：目录只存 文件名->inode号 的映射；数据块的去向全记在 inode 里。")
