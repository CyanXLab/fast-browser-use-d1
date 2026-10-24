<div align="center">

# ⚡ fast-browser-use-d1

**让 3B 小模型在纯 CPU 上驱动真实浏览器**
**单次前向决策 · 毫秒级 · 零 GPU · 每一步都可解释**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#-快速开始)
[![Inference](https://img.shields.io/badge/llama.cpp-CPU%20only-orange.svg)](#-工作原理)
[![Model](https://img.shields.io/badge/Model-LiquidAI%20d1--3B-purple.svg)](https://huggingface.co/LiquidAI)

[English Full Docs](README_EN.md) · [中文深度文档](README.zh-CN.md) · [CPU 实测报告](docs/d1-cpu-test-report.md)

</div>

---

## 这是什么

`fast-browser-use-d1` 是 [APUS-AI-Lab/fast-browser-use](https://github.com/APUS-AI-Lab/fast-browser-use) 的 **LiquidAI d1-3B + llama.cpp** 改造版。原项目用 LLM 给浏览器动作打分，本仓库把它改造成了一个**纯本地、可解释、能聊天**的浏览器智能体：

- 🧠 **决策模型换成 LiquidAI d1-3B**（System-1 决策模型）：所有候选动作**一次前向传播**同时打分，CPU（Q4_K_M 量化，1.7GB）就能实时跑，全程不依赖 GPU
- 💬 **Claude-Code 式交互 TUI**：`fbu chat` 打开可见浏览器，用自然语言下任务，实时看到每一步 d1 在想什么
- 🔭 **决策过程全透明**：每步显示 Top 候选概率、置信度、决策/执行耗时拆分、打分轮数——详细但不刷屏
- 🤝 **远程 LLM 协作（由 d1 自主决定）**：`ASK_LLM` 是一个被正常打分的一级候选动作——页面问答、任务总结、索要选项、卡住救援，何时咨询云端 LLM 完全由本地小模型自己决定，LLM 只提议、d1 来拍板
- 📋 **自动任务拆解**：多步目标自动拆成子目标清单，逐个里程碑推进（Claude-Code task list 风格）
- 🩹 **自愈能力**：连续无变化时自动请教 LLM 诊断卡因并修正子目标续跑；上下文超限自动压缩重试；页面变化自动重新决策
- 🧷 **会话记忆**：跨任务记住聊过的轮次和最后页面，"接着上一个任务继续"自然可用，重启也不丢
- 🪟 **Windows 一键部署**：`deploy.bat` 双击搞定 venv、Chromium、llama.cpp、权重校验，结束直接进入聊天循环

## 🚀 快速开始

### Windows（推荐，一键）

```text
1. 克隆本仓库（或下载 zip 解压）
2. 把 d1-3B-Q4_K_M.gguf 放进 models\ 目录
   （从 ModelScope / HuggingFace 下载 LiquidAI/d1-3B-GGUF，约 1.7GB）
3. 双击 deploy.bat —— 自动完成一切，最后进入聊天循环
```

部署脚本会：创建虚拟环境 → 安装依赖与 Playwright Chromium → 校验权重 → 自动下载
预编译 llama.cpp（CPU 版，无需 CMake）→ 跑通一次真实浏览器验证 → `fbu chat`。

### Linux / macOS

```bash
./deploy.sh            # 全自动，包含 llama.cpp 编译
# 或手动：
uv sync && fbu install-browser && fbu download --source modelscope
fbu chat               # 可见浏览器交互；--headless 隐藏
```

### 跑第一个任务

```text
❯ 打开 https://example.com 然后点击 More information 链接
⏺  1  A browser is a choice, not a conversation   conf 0.94 · 16.8s
   │ click「← Reading room」 95% · BLOCKED 2% · click「forma.」 2%   决策 16.2s · 执行 0.6s · 614 tok · 2 轮打分
⏺  2  DONE（判定子目标完成）   conf 0.73 · 8.0s
   │ DONE 73% · CONTINUE 27%   决策 7.9s · 607 tok
✓ 子目标完成 …
✔ 任务完成
```

## ⚙️ 工作原理

传统 LLM Agent 每步要"生成一段思考文本"再决定动作；d1 是 System-1 决策模型——
把**所有候选动作**（页面每个可点元素 + 控制动作 + DONE/BLOCKED/ASK_LLM）作为
label token，llama.cpp 的 `/v1/systemone` 端点**一次前向传播**返回全部校准概率：

```text
候选: A click「搜索」   B click「More information」   C scroll   …   Z ASK_LLM
              │
        一次前向传播（零生成 token）
              ▼
校准概率:  A 0.12   B 0.41   C 0.17  …  Z 0.03   →  选 B 执行
```

- **完成判定**走 d1 原生 yes/no 头（noul），比让小模型生成 "DONE" 文本可靠得多
- **字段填值**用 goal 片段打分兜底，保证精确输入
- 3B 模型 + 单 token 打分 = 2 核 CPU 每步 8~17 秒（含浏览器观察），核数越多越快

## 💬 聊天命令（fbu chat）

| 命令 | 作用 |
| :--- | :--- |
| `<任务文本>` | 执行浏览器任务，例：`打开 bing 搜索今日新闻并总结` |
| `/open <url>` | 只导航，不执行任务 |
| `/history` | 查看 LLM 能看到的会话历史 |
| `/decide` | 开关每步 d1 决策细节（也可 `FBU_STEP_DETAIL=0`） |
| `/retry` | 重试上一条任务 |
| `/clear` | 清空持久会话记忆 |
| `/llm` | 查看远程 LLM 协作状态 |
| `/exit` | 退出（Ctrl+C / Ctrl+D 也可以） |

## 🤝 什么时候会请云端 LLM？（不想要可以完全关掉）

- 页面问答 / "完成后总结" 类任务 → 最后一步 d1 主动选 `ASK_LLM`
- 置信度低于阈值 → 自动二次意见（LLM 的 BLOCKED 永远不被采纳，终止权在本地）
- 连续 3 步无页面变化 → LLM 诊断卡因，给出修正子目标自动续跑
- 任务里写"不要用 LLM" → d1 会顺应，自然不给 ASK_LLM 高分

`FBU_LLM_ESCALATION=off` = 100% 本地零云端调用；不配置 key 同样全程离线。

## 🔧 常用配置

完整列表见 [README.zh-CN.md](README.zh-CN.md) 或 `.env.example`：

| 环境变量 | 默认 | 说明 |
| :--- | :--- | :--- |
| `FBU_BACKEND` | `llamacpp` | `auto`/`llamacpp`/`mlx`/`torch` |
| `FBU_LLAMACPP_CTX` | `16384` | llama-server 上下文窗口，超限自动压缩重试 |
| `FBU_STEP_DETAIL` | `1` | 每步决策细节展示开关 |
| `FBU_MAX_CANDIDATES` | `28` | 决策候选表上限：大页面（如 bing）自动砍掉与目标无关的导航链接，提升置信度并降低 CPU 耗时 |
| `FBU_NO_VALUE_LIMIT` | `3` | 连续 N 次无法确定输入文本即停止，绝不把模型拒绝语（如 “None of the above…”）打进页面 |
| `FBU_PLAN` | `auto` | 多步目标自动拆解子目标清单 |
| `FBU_MAX_STEPS` | `60` | 每任务最大浏览器动作数 |
| `FBU_ASK_LLM_LIMIT` | `2` | 连续 LLM 咨询上限（防空转） |
| `FBU_LLM_ESCALATION` | `auto` | `auto`/`on`/`off`（off=纯本地） |
| `FBU_LLM_API_BASE` / `FBU_LLM_MODEL` / `FBU_LLM_API_KEY` | ModelScope 内置默认 | OpenAI 兼容远程 LLM，均可替换 |

## 📊 实测（2 核 CPU 沙箱，Q4_K_M）

| 任务 | 结果 |
| :--- | :--- |
| example.com → 点击 More information → IANA | ✅ PASS（精确 URL + 标题断言） |
| 本地多字段表单（输入/下拉/勾选/保存） | ✅ PASS 全对 |
| 多步新闻站导航 + 自动 6 步计划 + LLM 救援自愈 | ✅ PASS |
| 登录表单执行 | ✅ 执行成功（"元素消失即完成"判定为 3B 边界） |
| 会话恢复（重启后仅凭记忆答上一任务） | ✅ PASS |

详细数据与边界分析见 [docs/d1-cpu-test-report.md](docs/d1-cpu-test-report.md)。

## 🆚 与上游的主要差异

| | 上游 fast-browser-use | 本仓库 |
| :--- | :--- | :--- |
| 决策模型 | Qwen（需要 GPU/较大内存） | **LiquidAI d1-3B（Q4_K_M 量化，1.7GB）** |
| 推理 | PyTorch / MLX | **llama.cpp 纯 CPU，自动拉起 llama-server** |
| 交互 | CLI 一次性任务 | **Claude-Code 式 TUI 聊天循环 + 斜杠命令** |
| 决策透明度 | 内部 JSON trace | **每步实时渲染候选概率与耗时** |
| 云端 LLM 协作 | 无 | **d1 自主决定何时咨询，提议/决策分离** |
| 任务规划 | 单目标 | **自动子目标拆解 + 里程碑 + 卡住自愈** |
| 会话记忆 | 无 | **跨任务跨重启持久记忆** |
| 平台适配 | Linux/macOS 为主 | **Windows 一键部署（GBK 编码修复等）** |

## 🙏 致谢

- [APUS-AI-Lab/fast-browser-use](https://github.com/APUS-AI-Lab/fast-browser-use) — 核心范式（单次前向动作打分）与项目骨架
- [LiquidAI](https://huggingface.co/LiquidAI) — d1-3B System-1 决策模型
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — CPU 推理引擎
- [browser-use](https://github.com/browser-use/browser-use) — LLM 协作与回退思路来源

## License

[MIT](LICENSE)（继承上游）。模型权重遵循 LiquidAI 官方许可，请自行下载。
