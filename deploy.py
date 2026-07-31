#!/usr/bin/env python3
"""
deploy.py — 一键把本仓库发版到 GitHub Pages（经 GitHub REST API）

为什么不用 `git push`：本机 github.com:443 被防火墙掐，只有 api.github.com 可达，
所以改用 Git Database API 推送：blob -> tree -> commit -> 更新 main 引用。
PAT 运行时从 ~/.git-credentials 首行 `https://user:TOKEN@github.com` 读取，不写进仓库。

用法：
    python deploy.py                 # 默认提交信息 "deploy: <时间>"
    python deploy.py "自定义说明"     # 自定义提交信息
"""
import base64, datetime, json, os, re, subprocess, sys
import urllib.request, urllib.error

REPO = "Max-Well6/ele-basic-lecture"
API = "https://api.github.com"
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CREDS = os.path.expanduser(r"~/.git-credentials")
BRANCH = "main"


def read_token():
    with open(CREDS, "r", encoding="utf-8") as f:
        raw = f.read().replace("\r", "")
    return re.match(r"https?://[^/:]+:([^@]+)@.*", raw.splitlines()[0]).group(1).strip()


def api(method, path, data=None, accepts="application/vnd.github+json"):
    req = urllib.request.Request(API + path, method=method)
    req.add_header("Authorization", "Bearer " + TOKEN)
    req.add_header("Accept", accepts)
    req.add_header("User-Agent", "ele-basic-deploy")
    if data is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode("utf-8")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            body = r.read().decode("utf-8")
            return r.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")[:600]


TOKEN = read_token()

# 1) 列出全部受跟踪文件
files = [f for f in subprocess.check_output(["git", "ls-files"], cwd=REPO_DIR).decode().split() if f]
print(f"[files] {len(files)} tracked")

# 2) 取远程 main 当前顶端作为 parent；空仓库则先用 Contents API 初始化
st, resp = api("GET", f"/repos/{REPO}/git/ref/heads/{BRANCH}")
if st == 200 and isinstance(resp, dict) and "object" in resp:
    parent = resp["object"]["sha"]
    print(f"[parent] existing {BRANCH} = {parent}")
else:
    print("[init] 空仓库，用 Contents API 写 .gitignore 初始化")
    with open(os.path.join(REPO_DIR, ".gitignore"), "rb") as fh:
        gi = base64.b64encode(fh.read()).decode()
    st, resp = api("PUT", f"/repos/{REPO}/contents/.gitignore",
                   {"message": "init", "content": gi, "branch": BRANCH})
    if st not in (200, 201) or not isinstance(resp, dict):
        print(f"[INIT FAIL] {st} {resp}"); sys.exit(1)
    parent = resp["commit"]["sha"]
    print(f"[init] root = {parent}")

# 3) 逐文件建 blob
blobs = []
for fn in files:
    with open(os.path.join(REPO_DIR, fn), "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    st, resp = api("POST", f"/repos/{REPO}/git/blobs", {"content": b64, "encoding": "base64"})
    if st != 201 or not isinstance(resp, dict) or "sha" not in resp:
        print(f"[BLOB FAIL] {fn} {st} {resp}"); sys.exit(1)
    blobs.append({"path": fn, "mode": "100644", "type": "blob", "sha": resp["sha"]})
print(f"[blobs] created {len(blobs)}")

# 4) 建 tree（嵌套路径 GitHub 自动建中间 tree 对象）
st, resp = api("POST", f"/repos/{REPO}/git/trees", {"tree": blobs})
if st != 201 or not isinstance(resp, dict) or "sha" not in resp:
    print(f"[TREE FAIL] {st} {resp}"); sys.exit(1)
tree_sha = resp["sha"]

# 5) 建 commit（parent = 远程顶端，等价于快进）
msg = sys.argv[1] if len(sys.argv) > 1 else "deploy: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
st, resp = api("POST", f"/repos/{REPO}/git/commits",
               {"message": msg, "tree": tree_sha, "parents": [parent]})
if st != 201 or not isinstance(resp, dict) or "sha" not in resp:
    print(f"[COMMIT FAIL] {st} {resp}"); sys.exit(1)
commit_sha = resp["sha"]
print(f"[commit] {commit_sha}: {msg}")

# 6) 更新 main 引用（子提交，快进，无需 force）
st, resp = api("PATCH", f"/repos/{REPO}/git/refs/heads/{BRANCH}", {"sha": commit_sha})
if st != 200:
    print(f"[REF FAIL] {st} {resp}"); sys.exit(1)
print(f"[ref] {BRANCH} -> {commit_sha}")

# 7) 确保 Pages 已开启（幂等）
st, resp = api("GET", f"/repos/{REPO}/pages")
if st != 200:
    st2, resp2 = api("POST", f"/repos/{REPO}/pages", {"source": {"branch": BRANCH, "path": "/docs"}})
    print(f"[pages] enabled status={st2} -> {resp2 if isinstance(resp2, dict) else resp2}")
else:
    print(f"[pages] already enabled: {resp.get('html_url')}")
print("DONE")
