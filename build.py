# -*- coding: utf-8 -*-
"""
讲义构建脚本：将 docs/md/*.md 转换为交互式 HTML 讲义。

功能:
  - 侧边栏目录 / 顶栏课程导航
  - 明暗主题切换（记忆偏好，跟随系统）
  - 全站搜索（Ctrl+K，跨课程标题检索）
  - 代码块一键复制
  - Python 代码块浏览器内运行（Pyodide，按需加载）
  - Mermaid 图表渲染
  - 阅读进度条
  - 增量构建（按 mtime 跳过未修改文件）

用法:
    python build.py            # 增量构建
    python build.py --force    # 全量重建
扩展知识点:
    编辑 docs/md/*.md 后重新运行即可；新增课程在 SUBJECTS 注册表加一行。
依赖: pip install markdown pygments
"""
from __future__ import annotations

import ast
import html as htmllib
import json
import re
import sys
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).parent
MD_DIR = ROOT / "docs" / "md"
HTML_DIR = ROOT / "docs" / "html"

# 课程注册表：新增课程只需在此加一行并放入同名 md 文件
SUBJECTS = [
    ("01-digital", "数字电子技术", "门电路 · 组合/时序逻辑 · FSM · FPGA", "&#9881;"),
    ("02-analog", "模拟电子技术", "二极管 · 三极管 · 运放 · 反馈 · 电源", "&#9889;"),
    ("03-organization", "计算机组成原理", "数据表示 · CPU · 存储层次 · 流水线", "&#128421;"),
    ("04-architecture", "计算机体系结构", "量化设计 · ILP/DLP/TLP · DSA/AI芯片", "&#127959;"),
    ("05-data-structures", "数据结构", "链表 · 树 · 堆 · 哈希 · 图", "&#127794;"),
    ("06-algorithms", "算法设计与分析", "分治 · 贪心 · 动态规划 · 图算法 · NP", "&#129504;"),
    ("07-os", "操作系统", "进程 · 调度 · 内存 · 文件系统 · 并发", "&#128187;"),
]

PYGMENTS_LIGHT = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
PYGMENTS_DARK = HtmlFormatter(style="monokai").get_style_defs(
    'html[data-theme="dark"] .codehilite'
)

BASE_CSS = """
:root {
  --bg: #f7f8fa; --panel: #ffffff; --text: #24292f; --muted: #57606a;
  --accent: #0969da; --accent2: #8250df; --border: #d0d7de;
  --code-bg: #f3f5f7; --inline-code: #b3003c; --hl: #eaf1fb;
  --topbar: linear-gradient(90deg, #0b2545, #13315c 60%, #1d4e89);
  --sidebar-w: 300px; --shadow: rgba(9,105,218,.12);
}
html[data-theme="dark"] {
  --bg: #0d1117; --panel: #161b22; --text: #e6edf3; --muted: #9198a1;
  --accent: #58a6ff; --accent2: #bc8cff; --border: #30363d;
  --code-bg: #1c2128; --inline-code: #ff7b9c; --hl: #1c2b45;
  --topbar: linear-gradient(90deg, #010409, #0d1117 60%, #161b22);
  --shadow: rgba(0,0,0,.5);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
  line-height: 1.75; font-size: 16px;
  transition: background .2s, color .2s;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ---------- 阅读进度条 ---------- */
#progress {
  position: fixed; top: 0; left: 0; height: 3px; width: 0;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  z-index: 200; transition: width .1s linear;
}

/* ---------- 顶栏 ---------- */
.topbar {
  position: fixed; top: 0; left: 0; right: 0; height: 52px; z-index: 100;
  background: var(--topbar);
  color: #fff; display: flex; align-items: center; padding: 0 16px; gap: 14px;
  box-shadow: 0 1px 6px rgba(0,0,0,.25);
}
.topbar .brand { font-weight: 700; font-size: 15px; color: #fff; white-space: nowrap; }
.topbar nav { display: flex; gap: 3px; overflow-x: auto; flex: 1; scrollbar-width: none; }
.topbar nav::-webkit-scrollbar { display: none; }
.topbar nav a {
  color: #cfe1ff; padding: 5px 9px; border-radius: 6px; font-size: 13px; white-space: nowrap;
}
.topbar nav a:hover { background: rgba(255,255,255,.14); text-decoration: none; }
.topbar nav a.active { background: rgba(255,255,255,.22); color: #fff; font-weight: 600; }
.tb-btn {
  background: rgba(255,255,255,.12); border: none; color: #fff; cursor: pointer;
  padding: 6px 10px; border-radius: 6px; font-size: 13px; white-space: nowrap;
  display: flex; align-items: center; gap: 5px;
}
.tb-btn:hover { background: rgba(255,255,255,.24); }

/* ---------- 搜索面板 ---------- */
#searchMask {
  display: none; position: fixed; inset: 0; background: rgba(0,0,0,.45);
  z-index: 300; backdrop-filter: blur(2px);
}
#searchMask.open { display: block; }
#searchBox {
  max-width: 640px; margin: 90px auto 0; background: var(--panel);
  border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
  box-shadow: 0 20px 50px rgba(0,0,0,.35);
}
#searchInput {
  width: 100%; border: none; outline: none; padding: 16px 20px; font-size: 16px;
  background: transparent; color: var(--text);
  border-bottom: 1px solid var(--border);
}
#searchResults { max-height: 52vh; overflow-y: auto; }
#searchResults .sr {
  display: block; padding: 10px 20px; border-bottom: 1px solid var(--border); color: var(--text);
}
#searchResults .sr:hover, #searchResults .sr.sel { background: var(--hl); text-decoration: none; }
#searchResults .sr .t { font-size: 14.5px; }
#searchResults .sr .c { font-size: 12px; color: var(--muted); }
#searchResults .empty { padding: 20px; color: var(--muted); font-size: 14px; text-align: center; }
.sr-tip { padding: 8px 20px; font-size: 12px; color: var(--muted); }

/* ---------- 布局 ---------- */
.layout { display: flex; margin-top: 52px; min-height: calc(100vh - 52px); }
.sidebar {
  width: var(--sidebar-w); flex: 0 0 var(--sidebar-w);
  position: sticky; top: 52px; height: calc(100vh - 52px); overflow-y: auto;
  background: var(--panel); border-right: 1px solid var(--border); padding: 18px 10px 40px;
}
.sidebar .toc-title { font-size: 12px; color: var(--muted); letter-spacing: 2px; padding: 0 10px 8px; }
.sidebar ul { list-style: none; margin: 0; padding-left: 0; }
.sidebar li { margin: 1px 0; }
.sidebar a {
  display: block; padding: 4px 10px; border-radius: 6px; color: var(--text);
  font-size: 13.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.sidebar a:hover { background: var(--hl); text-decoration: none; }
.sidebar a.active { background: var(--hl); color: var(--accent); font-weight: 600; }
.sidebar ul ul a { padding-left: 26px; color: var(--muted); font-size: 13px; }
.sidebar ul ul ul a { padding-left: 42px; font-size: 12.5px; }

.content { flex: 1; min-width: 0; padding: 34px 48px 90px; max-width: 980px; margin: 0 auto; }

/* ---------- 正文排版 ---------- */
.content h1 {
  font-size: 30px; border-bottom: 3px solid var(--accent); padding-bottom: 10px; margin-top: 0;
}
.content h2 {
  font-size: 23px; margin-top: 44px; padding: 8px 14px; border-left: 5px solid var(--accent);
  background: linear-gradient(90deg, var(--hl), transparent); border-radius: 4px;
}
.content h3 { font-size: 18px; margin-top: 28px; color: var(--accent); }
.content blockquote {
  margin: 14px 0; padding: 10px 16px; border-left: 4px solid var(--accent2);
  background: var(--code-bg); color: var(--muted); border-radius: 4px;
}
.content table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 14.5px; }
.content th, .content td { border: 1px solid var(--border); padding: 7px 12px; text-align: left; }
.content th { background: var(--hl); }
.content tr:nth-child(even) { background: var(--code-bg); }
.content img, .content svg { max-width: 100%; }
.content code {
  background: var(--code-bg); padding: 2px 6px; border-radius: 4px;
  font-family: Consolas, "JetBrains Mono", Menlo, monospace; font-size: 87%;
  color: var(--inline-code);
}
hr { border: none; border-top: 1px solid var(--border); margin: 30px 0; }

/* ---------- 折叠答案 ---------- */
.content details {
  margin: 10px 0 18px; border: 1px solid var(--border); border-radius: 8px;
  background: var(--panel); padding: 0 16px;
}
.content details[open] { padding-bottom: 8px; }
.content details > summary {
  cursor: pointer; padding: 10px 0; font-weight: 600; color: var(--accent);
  list-style: none; user-select: none;
}
.content details > summary::before { content: "▸ "; }
.content details[open] > summary::before { content: "▾ "; }
.content details > summary::-webkit-details-marker { display: none; }

/* ---------- 代码块 ---------- */
.codewrap { position: relative; margin: 14px 0; }
.codehilite {
  background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px;
  padding: 2px 14px; overflow-x: auto; font-size: 13.5px; margin: 0;
}
.codehilite pre { margin: 10px 0; }
.codehilite code { background: transparent; padding: 0; color: inherit; font-size: 100%; }
.code-tools {
  position: absolute; top: 6px; right: 8px; display: flex; gap: 6px; opacity: 0;
  transition: opacity .15s;
}
.codewrap:hover .code-tools { opacity: 1; }
.code-tools button {
  background: var(--panel); border: 1px solid var(--border); color: var(--muted);
  font-size: 11.5px; padding: 3px 9px; border-radius: 5px; cursor: pointer;
}
.code-tools button:hover { color: var(--accent); border-color: var(--accent); }
.code-tools button.blocked {
  cursor: not-allowed; opacity: .55; border-style: dashed;
}
.code-tools button.blocked:hover { color: var(--muted); border-color: var(--border); }
.run-out {
  display: none; background: #0b1020; color: #b8f2c9; border: 1px solid var(--border);
  border-top: none; border-radius: 0 0 8px 8px; padding: 10px 14px; font-size: 12.5px;
  font-family: Consolas, monospace; white-space: pre-wrap; max-height: 340px; overflow: auto;
}
.run-out.show { display: block; }
.run-out.err { color: #ff9f9f; }

/* ---------- Mermaid ---------- */
.mermaid {
  background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; margin: 16px 0; text-align: center; overflow-x: auto;
}

.footer { text-align: center; color: var(--muted); font-size: 12.5px; padding: 26px 0 40px; }

/* ---------- 首页 ---------- */
.hero { text-align: center; padding: 46px 20px 8px; }
.hero h1 { font-size: 34px; border: none; margin-bottom: 6px; }
.hero p { color: var(--muted); max-width: 720px; margin: 8px auto; }
.stats { display: flex; justify-content: center; gap: 34px; margin: 22px 0 4px; flex-wrap: wrap; }
.stats div { text-align: center; }
.stats b { display: block; font-size: 26px; color: var(--accent); }
.stats span { font-size: 12.5px; color: var(--muted); }
.cards {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 18px; padding: 26px 40px 10px; max-width: 1160px; margin: 0 auto;
}
.card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 20px 22px; transition: .15s; display: block; color: var(--text);
}
.card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px var(--shadow);
  border-color: var(--accent); text-decoration: none; }
.card .icon { font-size: 26px; }
.card h3 { margin: 8px 0 4px; font-size: 17px; }
.card p { margin: 0; color: var(--muted); font-size: 13px; }
.notice {
  max-width: 900px; margin: 26px auto; background: var(--panel);
  border: 1px solid var(--border); border-radius: 12px; padding: 18px 26px; font-size: 14px;
}
.notice h3 { margin-top: 6px; }
.notice code { background: var(--code-bg); padding: 2px 6px; border-radius: 4px;
  color: var(--inline-code); font-family: Consolas, monospace; }

@media (max-width: 900px) {
  .sidebar { display: none; }
  .content { padding: 24px 18px 60px; }
  .topbar .brand { display: none; }
}
"""

# ---------- 前端脚本 ----------
COMMON_JS = r"""
// ===== 主题切换 =====
(function () {
  var saved = localStorage.getItem('lec-theme');
  var sysDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.setAttribute('data-theme', saved || (sysDark ? 'dark' : 'light'));
})();
function toggleTheme() {
  var cur = document.documentElement.getAttribute('data-theme');
  var next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('lec-theme', next);
  var b = document.getElementById('themeBtn');
  if (b) b.textContent = next === 'dark' ? '\u2600 亮色' : '\u263D 暗色';
  if (window.mermaid && document.querySelector('.mermaid')) location.reload();
}
document.addEventListener('DOMContentLoaded', function () {
  var b = document.getElementById('themeBtn');
  if (b) b.textContent = document.documentElement.getAttribute('data-theme') === 'dark'
    ? '\u2600 亮色' : '\u263D 暗色';
});

// ===== 全站搜索 =====
var SEARCH_IDX = __SEARCH_INDEX__;
var srSel = 0, srItems = [];
function openSearch() {
  document.getElementById('searchMask').classList.add('open');
  var i = document.getElementById('searchInput');
  i.value = ''; i.focus(); renderSR('');
}
function closeSearch() { document.getElementById('searchMask').classList.remove('open'); }
function renderSR(q) {
  var box = document.getElementById('searchResults');
  q = q.trim().toLowerCase();
  if (!q) {
    box.innerHTML = '<div class="sr-tip">输入关键词检索全部课程章节 · \u2191\u2193 选择 · Enter 跳转 · Esc 关闭</div>';
    srItems = []; return;
  }
  var hits = SEARCH_IDX.filter(function (r) {
    return r.t.toLowerCase().indexOf(q) >= 0 || r.c.toLowerCase().indexOf(q) >= 0;
  }).slice(0, 40);
  if (!hits.length) { box.innerHTML = '<div class="empty">没有找到匹配的章节</div>'; srItems = []; return; }
  box.innerHTML = hits.map(function (r, i) {
    return '<a class="sr' + (i === 0 ? ' sel' : '') + '" href="' + r.u + '">' +
      '<div class="t">' + r.t + '</div><div class="c">' + r.c + '</div></a>';
  }).join('');
  srSel = 0; srItems = box.querySelectorAll('.sr');
}
document.addEventListener('keydown', function (e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') { e.preventDefault(); openSearch(); return; }
  var mask = document.getElementById('searchMask');
  if (!mask || !mask.classList.contains('open')) return;
  if (e.key === 'Escape') { closeSearch(); }
  else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    if (!srItems.length) return;
    srItems[srSel].classList.remove('sel');
    srSel = (srSel + (e.key === 'ArrowDown' ? 1 : srItems.length - 1)) % srItems.length;
    srItems[srSel].classList.add('sel');
    srItems[srSel].scrollIntoView({ block: 'nearest' });
  } else if (e.key === 'Enter') {
    if (srItems.length) { e.preventDefault(); location.href = srItems[srSel].getAttribute('href'); }
  }
});

// ===== 代码复制 / 运行 =====
function copyCode(btn) {
  var pre = btn.closest('.codewrap').querySelector('pre');
  navigator.clipboard.writeText(pre.innerText).then(function () {
    var o = btn.textContent; btn.textContent = '\u2713 已复制';
    setTimeout(function () { btn.textContent = o; }, 1500);
  });
}
var pyodideReady = null;
function loadPyodideOnce(statusCb) {
  if (pyodideReady) return pyodideReady;
  statusCb('正在加载 Python 运行时（首次约 10 秒）...');
  pyodideReady = new Promise(function (resolve, reject) {
    var s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js';
    s.onload = function () {
      loadPyodide({ indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/' })
        .then(resolve).catch(reject);
    };
    s.onerror = function () { reject(new Error('Pyodide 加载失败，请检查网络连接')); };
    document.head.appendChild(s);
  });
  return pyodideReady;
}
function runCode(btn, slow) {
  var wrap = btn.closest('.codewrap');
  // 输出区是 .codewrap 的直接兄弟节点，必须按相邻关系取，
  // 不能用 parentNode.querySelector（那样所有代码块都会写进页面第一个输出区）
  var out = wrap.nextElementSibling;
  if (!out || !out.classList.contains('run-out')) return;
  if (slow && !btn.dataset.confirmed) {
    if (!confirm('这段代码包含性能测试或大规模循环。\n' +
                 '浏览器内的 Python 比本地慢数倍，运行期间页面会暂时无响应。\n\n仍要运行吗？')) return;
    btn.dataset.confirmed = '1';
  }
  var code = wrap.querySelector('pre').innerText;
  out.classList.add('show'); out.classList.remove('err');
  var setTxt = function (t) { out.textContent = t; };
  btn.disabled = true;
  loadPyodideOnce(setTxt).then(function (py) {
    setTxt('运行中 ...');
    var buf = [];
    py.setStdout({ batched: function (s) { buf.push(s); } });
    py.setStderr({ batched: function (s) { buf.push(s); } });
    // 让出一帧，确保"运行中"先渲染出来再进入同步执行
    setTimeout(function () {
      try {
        py.runPython(code);
        out.textContent = buf.join('\n') || '(无输出，程序正常结束)';
      } catch (err) {
        out.classList.add('err');
        out.textContent = (buf.length ? buf.join('\n') + '\n' : '') + String(err.message || err);
      }
      btn.disabled = false;
    }, 30);
  }).catch(function (e) {
    out.classList.add('err'); setTxt(String(e.message || e)); btn.disabled = false;
  });
}

// ===== 阅读进度 & 目录高亮 =====
document.addEventListener('scroll', function () {
  var h = document.documentElement;
  var p = h.scrollTop / (h.scrollHeight - h.clientHeight || 1) * 100;
  var bar = document.getElementById('progress');
  if (bar) bar.style.width = p + '%';
  var heads = document.querySelectorAll('.content h2, .content h3');
  var cur = null;
  for (var i = 0; i < heads.length; i++) {
    if (heads[i].getBoundingClientRect().top < 120) cur = heads[i].id;
  }
  if (cur) {
    document.querySelectorAll('.sidebar a').forEach(function (a) {
      a.classList.toggle('active', a.getAttribute('href') === '#' + cur);
    });
  }
}, { passive: true });
"""

MERMAID_JS = r"""
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
  if (window.mermaid) {
    mermaid.initialize({
      startOnLoad: true,
      theme: document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'default',
      securityLevel: 'loose'
    });
  }
</script>
"""

PAGE_TMPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · AI 时代计算机基础讲义</title>
<style>{base_css}
{pyg_light}
{pyg_dark}</style>
</head>
<body>
<div id="progress"></div>
<header class="topbar">
  <span class="brand">&#128218; AI 时代计算机基础</span>
  <nav>
    <a href="index.html">首页</a>
    {navlinks}
  </nav>
  <button class="tb-btn" onclick="openSearch()">&#128269; 搜索 <kbd>Ctrl K</kbd></button>
  <button class="tb-btn" id="themeBtn" onclick="toggleTheme()">&#9789; 暗色</button>
</header>
<div id="searchMask" onclick="if(event.target===this)closeSearch()">
  <div id="searchBox">
    <input id="searchInput" placeholder="搜索章节标题或课程名 ..." oninput="renderSR(this.value)" autocomplete="off">
    <div id="searchResults"></div>
  </div>
</div>
<div class="layout">
  <aside class="sidebar">
    <div class="toc-title">本页目录</div>
    {toc}
  </aside>
  <main class="content">
    {body}
    <div class="footer">AI 时代计算机基础讲义 · 源文件 docs/md/{stem}.md · 修改后运行 build.py 重新生成</div>
  </main>
</div>
<script>{common_js}</script>
{mermaid}
</body>
</html>
"""

INDEX_TMPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI 时代计算机基础讲义 · 总目录</title>
<style>{base_css}</style>
</head>
<body>
<div id="progress"></div>
<header class="topbar">
  <span class="brand">&#128218; AI 时代计算机基础</span>
  <nav>
    <a class="active" href="index.html">首页</a>
    {navlinks}
  </nav>
  <button class="tb-btn" onclick="openSearch()">&#128269; 搜索 <kbd>Ctrl K</kbd></button>
  <button class="tb-btn" id="themeBtn" onclick="toggleTheme()">&#9789; 暗色</button>
</header>
<div id="searchMask" onclick="if(event.target===this)closeSearch()">
  <div id="searchBox">
    <input id="searchInput" placeholder="搜索章节标题或课程名 ..." oninput="renderSR(this.value)" autocomplete="off">
    <div id="searchResults"></div>
  </div>
</div>
<div class="hero">
  <h1>AI 时代计算机基础讲义</h1>
  <p>面向本科生的核心课程体系：从晶体管到操作系统，理解 AI 算力世界的完整底座。
  每门课配<b>可视化图示</b>、<b>可运行案例代码</b>、<b>折叠式习题答案</b>与<b>可扩展知识点</b>。</p>
  <div class="stats">
    <div><b>{n_sub}</b><span>门课程</span></div>
    <div><b>{n_line}</b><span>行讲义</span></div>
    <div><b>{n_code}</b><span>个示例代码</span></div>
    <div><b>{n_ex}</b><span>道习题</span></div>
  </div>
</div>
<div class="cards">
  {cards}
</div>
<div class="notice">
  <h3>&#128640; 如何使用</h3>
  <ul>
    <li><b>阅读</b>：点击卡片进入课程，左侧为章节目录；右上角可切换<b>明暗主题</b>，<code>Ctrl+K</code> 全站搜索。</li>
    <li><b>动手</b>：Python 代码块右上角有<b>「运行」</b>按钮，可直接在浏览器里执行（Pyodide）；也可用<b>「复制」</b>拿到本地跑。</li>
    <li><b>练习</b>：每门课末尾习题带折叠式参考答案，点击展开。</li>
    <li><b>扩展</b>：源文件在 <code>docs/md/</code>；修改后运行 <code>python build.py</code> 增量重建（<code>--force</code> 全量）。</li>
    <li><b>本地示例</b>：<code>code/&lt;课程&gt;/</code> 目录下为独立可运行脚本，Python 示例零依赖。</li>
  </ul>
</div>
<div class="footer">MD 源文件与 HTML 双格式 · 构建工具 build.py</div>
<script>{common_js}</script>
</body>
</html>
"""


def nav_links(active_stem: str | None) -> str:
    out = []
    for stem, title, _, _ in SUBJECTS:
        if not (MD_DIR / f"{stem}.md").exists():
            continue
        cls = ' class="active"' if stem == active_stem else ""
        out.append(f'<a{cls} href="{stem}.html">{title}</a>')
    return "\n    ".join(out)


# Pyodide（浏览器内 CPython/WASM）不支持的标准库：无操作系统线程、无网络套接字、无子进程
PYODIDE_BLOCKED = {
    "threading": "多线程",
    "multiprocessing": "多进程",
    "subprocess": "子进程",
    "socket": "网络套接字",
    "concurrent": "线程/进程池",
    "ctypes": "本地库调用",
    "tkinter": "GUI",
    "input": "交互式输入",
}
# 计时基准与大规模循环：Pyodide 比原生慢 3-20 倍，同步执行会冻结页面
SLOW_PAT = re.compile(r"perf_counter|time\.time\(\)|process_time|range\(\s*\d{5,}")
# 判定为 Python 的最低结构要求，避免把 ```text 输出示例误认成代码
PY_NODES = (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom,
            ast.Assign, ast.For, ast.While, ast.If, ast.Call)


def classify_code(code: str) -> tuple[str, str]:
    """判定代码块能否在浏览器里运行。

    返回 (状态, 原因)：
      run   — 可直接运行
      slow  — 可运行但耗时，运行前需二次确认
      block — Pyodide 不支持，禁用运行按钮
      na    — 非 Python（Verilog / C / 纯文本），不提供运行按钮
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "na", ""
    if not any(isinstance(n, PY_NODES) for n in ast.walk(tree)):
        return "na", ""

    hits = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            hits.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "input":
                hits.add("input")
    bad = sorted(hits & PYODIDE_BLOCKED.keys())
    if bad:
        return "block", "、".join(PYODIDE_BLOCKED[b] for b in bad)
    if SLOW_PAT.search(code):
        return "slow", ""
    return "run", ""


FENCE_RE = re.compile(r"^[ \t]*```+[ \t]*([A-Za-z0-9_+-]*)", re.M)


def fenced_langs(md_text: str) -> list[str]:
    """按文档顺序提取围栏代码块的语言标签。

    mermaid 块在 fix_mermaid 中已被替换为 <div class="mermaid">，
    不会进入 enhance_code_blocks，因此这里剔除，保证序号一一对应。
    """
    langs, opened = [], False
    for m in FENCE_RE.finditer(md_text):
        if opened:          # 闭栏
            opened = False
            continue
        langs.append(m.group(1).lower())
        opened = True
    return [x for x in langs if x != "mermaid"]


def enhance_code_blocks(html: str, stats: dict | None = None,
                        langs: list[str] | None = None) -> str:
    """给代码块套工具栏；可运行的 Python 块额外带「运行」按钮和输出区。

    langs 为按顺序排列的围栏语言标签。给出时以其为准（精确），
    为 None 时退回按 AST 结构猜测（可能把恰好合法的文本误判为 Python）。
    """
    pattern = re.compile(r'<div class="codehilite">(.*?)</div>', re.S)
    lang_iter = iter(langs) if langs is not None else None

    def repl(m: re.Match) -> str:
        inner = m.group(0)
        code = htmllib.unescape(re.sub(r"<[^>]+>", "", inner))
        lang = next(lang_iter, "") if lang_iter is not None else None
        if lang is not None and lang not in ("python", "py", "python3"):
            state, reason = "na", ""
        else:
            state, reason = classify_code(code)
        if stats is not None:
            stats[state] = stats.get(state, 0) + 1

        run_btn, out_div = "", ""
        if state == "run":
            run_btn = ('<button onclick="runCode(this,0)" '
                       'title="在浏览器中运行">&#9654; 运行</button>')
            out_div = '<div class="run-out"></div>'
        elif state == "slow":
            run_btn = ('<button onclick="runCode(this,1)" '
                       'title="可运行，但含性能测试或大规模循环，浏览器中较慢">'
                       '&#9654; 运行 &#9203;</button>')
            out_div = '<div class="run-out"></div>'
        elif state == "block":
            run_btn = (f'<button class="blocked" disabled '
                       f'title="浏览器 Python（Pyodide）不支持{reason}，'
                       f'请下载到本地运行">&#9654; 仅本地可运行</button>')

        return (
            f'<div class="codewrap">{inner}'
            f'<div class="code-tools">{run_btn}'
            f'<button onclick="copyCode(this)">&#128203; 复制</button></div>'
            f'</div>{out_div}'
        )

    return pattern.sub(repl, html)


def fix_mermaid(html: str) -> str:
    """把 ```mermaid 生成的代码块还原为 <div class="mermaid"> 供前端渲染。"""
    import html as htmllib

    def repl(m: re.Match) -> str:
        raw = re.sub(r"<[^>]+>", "", m.group(1))
        return f'<div class="mermaid">{htmllib.unescape(raw)}</div>'

    html = re.sub(
        r'<div class="codehilite"><pre><span></span><code>(.*?)</code></pre></div>',
        lambda m: (
            repl(m)
            if re.match(r"\s*(graph|flowchart|stateDiagram|sequenceDiagram|classDiagram|erDiagram|gantt|pie|journey)",
                        re.sub(r"<[^>]+>", "", m.group(1)))
            else m.group(0)
        ),
        html,
        flags=re.S,
    )
    return html


def make_md() -> markdown.Markdown:
    return markdown.Markdown(
        extensions=["extra", "toc", "codehilite", "sane_lists", "admonition", "md_in_html"],
        extension_configs={
            "toc": {"toc_depth": "2-3", "permalink": False},
            "codehilite": {"guess_lang": False, "noclasses": False},
        },
    )


def build_page(stem: str, title: str, search_idx: list) -> tuple[bool, int]:
    src = MD_DIR / f"{stem}.md"
    if not src.exists():
        print(f"  [跳过] {src.name} 不存在")
        return False, 0
    text = src.read_text(encoding="utf-8")
    md = make_md()
    body = md.convert(text)
    body = fix_mermaid(body)
    code_stats: dict[str, int] = {}
    # 优先用 MD 源码里的语言标签精确判定；数量对不上（缩进式代码块等）时回退到结构猜测
    langs = fenced_langs(text)
    if len(langs) != len(re.findall(r'<div class="codehilite">', body)):
        print(f"  [提示] {stem}: 围栏标签 {len(langs)} 个与代码块数不符，回退结构判定")
        langs = None
    body = enhance_code_blocks(body, code_stats, langs)
    toc = md.toc or "<ul></ul>"

    # 收集搜索索引（二级/三级标题）
    for lvl, tid, name in re.findall(
        r'<h([23]) id="([^"]+)">(.*?)</h[23]>', body, re.S
    ):
        clean = re.sub(r"<[^>]+>", "", name).strip()
        if clean:
            search_idx.append({"t": clean, "c": title, "u": f"{stem}.html#{tid}"})

    common = COMMON_JS.replace("__SEARCH_INDEX__", "__IDX__")
    html = PAGE_TMPL.format(
        title=title, base_css=BASE_CSS, pyg_light=PYGMENTS_LIGHT, pyg_dark=PYGMENTS_DARK,
        navlinks=nav_links(stem), toc=toc, body=body, stem=stem,
        common_js=common, mermaid=MERMAID_JS if 'class="mermaid"' in body else "",
    )
    (HTML_DIR / f"{stem}.html").write_text(html, encoding="utf-8")
    n_lines = len(text.splitlines())
    runnable = code_stats.get("run", 0) + code_stats.get("slow", 0)
    extra = f"可运行 {runnable}"
    if code_stats.get("slow"):
        extra += f"(慢 {code_stats['slow']})"
    if code_stats.get("block"):
        extra += f" 禁用 {code_stats['block']}"
    print(f"  [OK] {stem}.html  ({n_lines} 行 -> {len(html)//1024} KB, {extra}"
          f"{', 含 Mermaid' if 'class=\"mermaid\"' in body else ''})")
    return True, n_lines


def count_assets() -> tuple[int, int]:
    code_dir = ROOT / "code"
    n_code = len([p for p in code_dir.rglob("*") if p.is_file()]) if code_dir.exists() else 0
    n_ex = 0
    for p in MD_DIR.glob("*.md"):
        n_ex += len(re.findall(r"<summary>参考答案</summary>", p.read_text(encoding="utf-8")))
    return n_code, n_ex


def build_index(total_lines: int, search_idx: list) -> None:
    cards = []
    n_sub = 0
    for stem, title, desc, icon in SUBJECTS:
        if not (MD_DIR / f"{stem}.md").exists():
            continue
        n_sub += 1
        cards.append(
            f'<a class="card" href="{stem}.html">'
            f'<div class="icon">{icon}</div><h3>{title}</h3><p>{desc}</p></a>'
        )
    n_code, n_ex = count_assets()
    common = COMMON_JS.replace("__SEARCH_INDEX__", "__IDX__")
    html = INDEX_TMPL.format(
        base_css=BASE_CSS, navlinks=nav_links(None), cards="\n  ".join(cards),
        common_js=common, n_sub=n_sub, n_line=f"{total_lines:,}", n_code=n_code,
        n_ex=n_ex or "—",
    )
    (HTML_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"  [OK] index.html  ({n_sub} 门 / {total_lines} 行 / {n_code} 代码 / {n_ex} 答案)")


def inject_search_index(search_idx: list) -> None:
    """所有页面生成后统一注入全站搜索索引。"""
    idx_json = json.dumps(search_idx, ensure_ascii=False)
    for p in HTML_DIR.glob("*.html"):
        t = p.read_text(encoding="utf-8")
        if "__IDX__" in t:
            p.write_text(t.replace("__IDX__", idx_json), encoding="utf-8")


def main() -> None:
    force = "--force" in sys.argv
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    print("构建 HTML 讲义 ..." + ("（全量）" if force else "（增量）"))

    search_idx: list = []
    total = 0
    built = 0
    for stem, title, _, _ in SUBJECTS:
        src = MD_DIR / f"{stem}.md"
        dst = HTML_DIR / f"{stem}.html"
        if not src.exists():
            continue
        stale = force or not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime
        if stale:
            ok, n = build_page(stem, title, search_idx)
            built += int(ok)
            total += n
        else:
            # 未修改也要重建索引（不写盘）
            md = make_md()
            body = md.convert(src.read_text(encoding="utf-8"))
            for _lvl, tid, name in re.findall(r'<h([23]) id="([^"]+)">(.*?)</h[23]>', body, re.S):
                clean = re.sub(r"<[^>]+>", "", name).strip()
                if clean:
                    search_idx.append({"t": clean, "c": title, "u": f"{stem}.html#{tid}"})
            total += len(src.read_text(encoding="utf-8").splitlines())
            print(f"  [跳过] {stem}.html 已是最新")

    build_index(total, search_idx)
    inject_search_index(search_idx)
    print(f"完成：重建 {built} 个页面，共 {total} 行讲义 -> {HTML_DIR}")


if __name__ == "__main__":
    main()
