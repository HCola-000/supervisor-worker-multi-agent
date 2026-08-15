# supervisor-worker-multi-agent

一个基于 [LangGraph](https://github.com/langchain-ai/langgraph) 的**主管-工作者（Supervisor-Worker）多智能体**应用：联网调研主题、自动整理资料、撰写 Markdown 文章，并支持对已写内容的追问和改稿。

## 工作流程

```
用户消息
   ↓
supervisor（意图识别 + 路由）
   ├── new_task → researcher（Tavily 搜索）→ summarizer（提炼摘要）→ writer（写 .md）
   ├── followup → answer（用已有资料/文章直接回答，不写文件）
   └── revise   → writer（read_file 读原文件 → 改 → save_file 保存）
```

## 功能特性

- 真实联网搜索（Tavily）
- 自动整理资料摘要
- 自动生成 Markdown 文件
- 追问答疑（基于已有资料回答，不重复写文件）
- 改稿（读取已有文件并修改）
- 流式输出

## 环境要求

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) 或 pip

## 安装

```bash
# 用 uv（推荐）
uv sync

# 或 pip
pip install -e .
```

## 配置

1. 复制环境变量模板：

   ```bash
   cp .env.example .env
   ```

2. 在 `.env` 里填入：
   - `BASE_URL` / `API_KEY`：DeepSeek API
   - `TAVILY_API_KEY`：Tavily 搜索 key（https://app.tavily.com）

3. （可选）在 `src/agent/graph.py` 顶部改 `OUTPUT_DIR`，指定文章的保存目录。

## 运行

端到端跑一遍（联网调研 → 写文件）：

```bash
python e2e_test.py
```

或用 LangGraph Studio 可视化调试（可在同一线程连续追问、改稿）：

```bash
langgraph dev --no-docker   # 免 Docker；或先启动 Docker 再 langgraph dev
```

## 项目结构

```
src/agent/
  ├── graph.py   # 主管-工作者图定义（意图识别、路由、各节点）
  └── state.py   # 共享状态
e2e_test.py          # 端到端测试脚本
smoke_test_search.py # Tavily 搜索冒烟测试
```

## License

[MIT](LICENSE)
