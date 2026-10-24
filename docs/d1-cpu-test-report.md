# d1-3B + llama.cpp CPU 推理 · 真实浏览器自动化测试报告

- 测试日期：2026-10-24
- 被测模型：**LiquidAI/d1-3B**（`d1-3B-Q4_K_M.gguf`，约 1.67 GB）
- 推理栈：llama.cpp v0.7.0 `llama-server`（源码编译，`GGML_NATIVE=ON`），决策走原生 `/v1/systemone` 端点
- 测试环境：2 vCPU / 4 GB 内存 Linux 沙箱（无 GPU、无浏览器桌面），headless Chromium (Playwright)
- 决策模式：`FBU_DECISION_MODE=binary`（d1 默认，独立完成判断通道，noul yes/no 头）

## 场景与结果

| # | 场景 | 任务 | 结果 | 执行动作（置信度） | 独立断言 | 总耗时 |
|---|------|------|------|--------------------|----------|--------|
| A | 真实网站 example.com | 打开 "More information" 链接进入 IANA | ✅ PASS | SCROLL_DOWN (0.53) → CLICK "Learn more" (0.90) → DONE | 精确 URL + 标题匹配 | 59.7 s |
| B | 本地表单 settings.html | 保存工作区偏好（名称/时区/周报/提交） | ✅ PASS | TYPE_TEXT "Orchard" (0.90) → SELECT Asia/Singapore (0.85) → CLICK 周报开关 (0.51) → CLICK Save (0.97) → DONE | 标题 + 3 项文本断言 | 74.6 s |
| C | 本地 fixture 多文章导航 | 打开指定文章 | ❌ FAIL | 置信度扩散 (0.15~0.66)，多次跳转后触及步数预算 | 未达成 | 402 s |

对应执行轨迹（原始 trace JSON）：`artifacts/trace_A2.json`、`artifacts/trace_B4.json`、`artifacts/trace_C.json`

## 关键发现

1. **`/v1/systemone` 与 d1 决策头完美配合**：动作选择单次前向完成（0 个生成 token），41 个候选项时仍能选出正确动作（97% 置信）。
2. **noul（yes/no）头显著优于 A/B choice 提问**：完成判断若用 choice 问法会误判（0.88 偏向"已完成"）；改用 noul 严格问法后校准极佳（0.07 = 明确未完成）。已将 noul 作为后端默认完成判断通道。
3. **TYPE_TEXT 字段填充**：chat 模板 + 贪婪解码可以直接产出 `{"text":"Orchard"}` 风格的精确抽取；另实现了基于 goal 片段打分的 System-1 兜底通道。
4. **延迟**：2 vCPU 上每步 15–23 s（prefill 主导）；8 核桌面预计 2–6 s/步。llama.cpp 的 systemone 决策不含自回归解码，延迟随核心数近线性下降。
5. **能力边界**：d1-3B 在多相似链接的区分任务上置信度分散（场景 C）；这类任务建议配合 `FBU_PLAN=1` 子目标拆解或更高量化档位（Q8_0/BF16）。

## 复现方式

```bash
bash deploy.sh                      # 一键部署（构建 llama.cpp + 下载权重 + 装浏览器 + 冒烟测试）
bash deploy.sh --skip-test          # 跳过冒烟测试
# 逐场景复现（权重与 llama-server 路径按 deploy.sh 输出调整）
FBU_MODEL=$PWD/models/d1-3B-Q4_K_M.gguf FBU_LLAMACPP_BIN=<llama-server> \
  uv run fbu run https://example.com \
  --goal 'Open the "More information" link so the IANA example-domains page is shown.' \
  --expect-url 'https://www.iana.org/help/example-domains' --expect-title 'Example Domains'
```

## 单元测试

`pytest -q`：**127 passed, 11 skipped**（skipped 为需 MLX/torch 权重的平台专项）；`ruff check` 全绿。

## 远程 LLM 协作（ASK_LLM）实测补充（2026-10-24）

新增 `fast_browser_use/remote_llm.py`：可选远程 LLM 顾问（默认魔搭 ModelScope
API-Inference，`deepseek-ai/DeepSeek-V4.1-Flash-1017`，低思考模式）。核心设计：
**ASK_LLM 作为一级候选动作进入 d1 的单次前向打分空间**，由决策模型自主决定何时
调用 LLM（页面问答 / 任务总结 / 低置信度二次意见 / 卡住诊断与子目标修正），
任务提示词可直接控制（"不用 LLM / 少用 LLM"会被 d1 顺应）。

| 场景 | 配置 | 结果 |
| :--- | :--- | :--- |
| 总结任务（example.com） | 默认 auto | d1 以 0.942/0.957 置信度主动选 ASK_LLM，LLM 正确总结页面，任务 done，`summary` 落入 trace |
| 导航任务（example.com → IANA） | 默认 auto | 断言通过，全程零 LLM 调用（d1 未给 ASK_LLM 高分） |
| 强制升级（每步 LLM 复核） | `FBU_LLM_CONFIDENCE=0.99` | 断言通过；升级记录完整写入 trace |
| ModelScope API 连通性 | 内置默认 key | `/v1/chat/completions` 返回正常，`enable_thinking:false` 被接受 |

工程护栏：
- 远程 LLM 的二次意见**永远无法覆盖为 BLOCKED**——终止决定权保留在本地决策模型；
- LLM 对卡住的诊断只能"提议"修正子目标，是否执行仍由 d1 重新打分决定；
- API 失败（超时/断网）只记录 escalation 错误，绝不中断任务；
- `FBU_LLM_API_KEY=""` 或 `FBU_LLM_ESCALATION=off` 可回到 100% 本地。

`pytest -q`：**141 passed, 11 skipped**（新增 tests/test_remote_llm.py 12 例 +
conftest 本地化夹具）；`ruff check` 全绿。Windows 一键部署：`deploy.bat`
（自动装依赖/Chromium/预编译 llama.cpp CPU 版/校验权重），权重位置
`models\d1-3B-Q4_K_M.gguf`。

## 会话记忆、TUI 与 Agent 循环强化实测（2026-10-24 第二批）

### LLM 会话记忆

聊天会话的每轮（任务 / LLM 回答 / 到达页面）会注入后续所有远程 LLM 调用的
`SESSION HISTORY` 块（限 6 轮 / 1800 字符），并持久化到
`artifacts/chat-session.json`（重启 `fbu chat` 自动恢复，`/clear` 清空）。

| 场景 | 结果 |
| :--- | :--- |
| 第二轮直接问"上一个任务总结的页面标题是什么" | LLM 零浏览器操作、仅凭 SESSION HISTORY 正确答出 "Example Domain" |
| 重启进程后续接任务 | 持久记忆恢复（轮次 + 上次 URL），`/history` 可查看 LLM 可见的历史 |

工程护栏：`FBU_ASK_LLM_LIMIT=2` 限制两次浏览器动作之间的连续咨询次数——实测曾出现
d1 连续 14 次选择 ASK_LLM 的空转，加护栏后第 3 次咨询即按"已有答案"收尾。

### 高难度长任务实测（books.toscrape.com / quotes.toscrape.com，2 核 CPU 沙箱）

| 任务 | 结果 |
| :--- | :--- |
| books：进 Travel 分类找最便宜的书（多步） | **PASS**：d1 自动生成 6 步计划（计划面板实时展示）；步骤 2-5 陷入点击 "Books" 头部链接后，卡住检测触发，LLM 救援精准诊断"重复点击不导航的头部链接而非侧边栏 Travel 链接"，自愈后直达 Travel 分类页（milestone/里程碑机制生效） |
| books 续轮：找最便宜的书（重启进程） | **持久记忆 PASS**（恢复轮次+URL），比价判断为 3B 能力边界（11 本书的价格逐个比较置信度扩散 0.12-0.44，与已知边界一致） |
| quotes：admin/admin 登录 + 提取名言 ×2 | 表单填写与提交实际执行成功（URL 跳转首页），但"子目标已完成"的停止判断为能力边界（元素消失后 d1 继续找 Login → 触发卡死判定 blocked）；两次 LLM 诊断均正确指出卡因 |
| 对话恢复：直接问结果 | **PASS**：续接会话 → d1 自主选 ASK_LLM → 正确答出第一句名言 "The world as we have created it is a process of our thinking..."（Albert Einstein） |

结论：多步导航 + 卡住自愈 + 会话续接 + LLM 协作问答链路完整可用；纯比较类
（多候选价格对比）与"负证据完成判定"（目标元素消失即完成）是 d1-3B 的能力边界，
对话式补指令（Claude-Code 风格）可自然绕过。

### 缓存命中实测

| 措施 | 实测结果 |
| :--- | :--- |
| `--cache-ram 1024 --cache-reuse 256` + `cache_prompt:true` | chat/plan/字段文本生成端点命中：同前缀二次请求 1.16s → 0.60s（**省 48%**） |
| 决策请求跨步 KV 复用 | **不可用**：verbose 日志确认 LFM2 混合/循环记忆触发 llama.cpp 的 "forcing full prompt re-processing due to lack of cache data (hybrid/recurrent memory)"（上游 PR #13194 限制），每步全量重算属架构级约束，参数已预置、未来 llama.cpp 支持后自动生效 |
| 请求内前缀共享（合并 choice+noul 双问题） | 分组机制确实触发（父/子任务），但当前 CPU 构建在分组路径存在段错误（libggml-cpu），已回退为独立请求 |
| 决策 prompt 去重 | 尾部与 GOAL/RECENT ACTIONS 重复的字段已移除，每步少算冗余 token |

### 测试

`pytest -q`：**147 passed, 11 skipped**（新增会话历史 6 例）；`ruff check` 全绿。
