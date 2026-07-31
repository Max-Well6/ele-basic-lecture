# AI 时代计算机基础讲义

面向本科生的六门核心课程讲义，**MD + HTML 双格式**，配可直接运行的案例代码。

## 课程列表

| # | 课程 | MD 源文件 | HTML | 示例代码 |
|---|------|-----------|------|----------|
| 1 | 数字电子技术 | `docs/md/01-digital.md` | `docs/html/01-digital.html` | `code/01-digital/` |
| 2 | 模拟电子技术 | `docs/md/02-analog.md` | `docs/html/02-analog.html` | `code/02-analog/` |
| 3 | 计算机组成原理 | `docs/md/03-organization.md` | `docs/html/03-organization.html` | `code/03-organization/` |
| 4 | 计算机体系结构 | `docs/md/04-architecture.md` | `docs/html/04-architecture.html` | `code/04-architecture/` |
| 5 | 数据结构 | `docs/md/05-data-structures.md` | `docs/html/05-data-structures.html` | `code/05-data-structures/` |
| 6 | 算法设计与分析 | `docs/md/06-algorithms.md` | `docs/html/06-algorithms.html` | `code/06-algorithms/` |

## 快速开始

```bash
# 阅读：浏览器打开总目录
docs/html/index.html

# 运行案例（Python 示例零依赖）
python code/03-organization/toy_cpu.py

# Verilog 示例（需 iverilog）
iverilog -o sim code/01-digital/adder.v && vvp sim
```

## 如何扩展知识点

1. 直接编辑 `docs/md/` 下对应课程的 Markdown 文件（每门课末尾附"扩展知识点"清单，可按图索骥补充章节）
2. 重新生成 HTML：

```bash
pip install markdown pygments   # 仅首次
python build.py
```

3. 新增整门课程：在 `build.py` 的 `SUBJECTS` 注册表加一行，放入同名 md 文件，重新构建即可。

## 讲义结构约定

每门课统一结构，便于横向对照与持续扩展：

- `## 0. AI 时代为什么还要学 X` —— 与 AI 算力/芯片/系统的联系
- `## 1~9` 正文章节 —— 知识要点 + 概念精讲 + 可运行案例代码
- `## 扩展知识点` —— 进阶方向清单（预留扩展入口）
- `## 练习与思考题` / `## 参考资料`
