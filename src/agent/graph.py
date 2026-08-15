"""LangGraph 主管-工作者多智能体：调研 → 总结 → 写文件，支持追问与改稿。"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from tavily import TavilyClient

from .state import State

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    base_url=BASE_URL,
    api_key=API_KEY,
)

# ---------- 工具 ----------

_tavily_client: TavilyClient | None = None
OUTPUT_DIR = "output"


def _get_tavily_client() -> TavilyClient | None:
    """返回 Tavily 客户端，未配置 API key 时返回 None。"""
    global _tavily_client
    if TAVILY_API_KEY is None:
        return None
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return _tavily_client


@tool
def search(query: str) -> str:
    """在互联网上搜索查询，返回相关网页的标题、链接和内容摘要。"""
    client = _get_tavily_client()
    if client is None:
        return "错误：未配置 TAVILY_API_KEY 环境变量，无法联网搜索。"
    response = client.search(query=query, max_results=5)
    results = response.get("results", [])
    if not results:
        return f"关于 '{query}' 没有找到相关结果。"
    return "\n".join(
        f"- {r.get('title', '无标题')} ({r.get('url', '')})\n  {r.get('content', '')}"
        for r in results
    )


@tool
def save_file(filename: str, content: str) -> str:
    """把内容保存到本地文件（output 目录下），返回保存的绝对路径。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = os.path.basename(filename) or "draft.md"
    if not safe_name.endswith(".md"):
        safe_name += ".md"
    path = os.path.join(OUTPUT_DIR, safe_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return os.path.abspath(path)


@tool
def read_file(filename: str) -> str:
    """读取 output 目录下指定文件的内容。"""
    path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        return f"文件不存在: {path}"
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------- 智能体 ----------

research_agent = create_agent(
    model=model,
    tools=[search],
    system_prompt=(
        "你是一名研究员。根据用户需求调用 search 工具查找资料，"
        "最后用中文输出结构化的调研结果：关键事实、要点和来源链接。"
    ),
)

write_agent = create_agent(
    model=model,
    tools=[save_file, read_file],
    system_prompt=(
        "你是一名专业写手。根据资料摘要写一篇结构完整、内容详实的 Markdown 文章，"
        "写完用 save_file 保存成 .md 文件（英文/拼音文件名）。"
        "如果要修改已有文章，先用 read_file 读取原文件，按要求修改后再 save_file 保存。"
    ),
)


def _last_ai_content(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not m.tool_calls:
            return m.content
    return ""


def _extract_saved_article(messages) -> tuple[str, str]:
    draft = ""
    path = ""
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for call in m.tool_calls:
                if call.get("name") == "save_file":
                    draft = call.get("args", {}).get("content", "")
        if getattr(m, "type", "") == "tool":
            path = m.content
    return draft, path


# ---------- 节点 ----------

def researcher_node(state: State) -> dict:
    input_msgs = state["messages"]
    result = research_agent.invoke({"messages": input_msgs})
    new_msgs = result["messages"][len(input_msgs):]
    return {
        "messages": new_msgs,
        "research_results": _last_ai_content(new_msgs),
    }


def summarizer_node(state: State) -> dict:
    research = state.get("research_results", "")
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "你是一名资料整理员。把调研结果提炼成结构化的要点摘要，"
                    "保留关键事实、数据和来源，去掉冗余，用于后续写文章。"
                )
            ),
            HumanMessage(content=research),
        ]
    )
    summary = response.content
    return {
        "messages": [AIMessage(content=f"【资料摘要】\n{summary}")],
        "research_summary": summary,
    }


def writer_node(state: State) -> dict:
    intent = state.get("intent", "new_task")
    if intent == "revise":
        instruction = state["messages"][-1].content
        file_path = state.get("file_path", "")
        prompt = (
            f"用户要修改已有文章。原文件路径：{file_path}\n"
            f"修改要求：{instruction}\n"
            "请先用 read_file 读取该文件内容，按要求修改，再用 save_file 保存。"
        )
    else:
        summary = state.get("research_summary", "")
        prompt = f"请根据下面这份资料摘要写文章并保存文件：\n\n{summary}"

    input_msgs = state["messages"] + [HumanMessage(content=prompt)]
    result = write_agent.invoke({"messages": input_msgs})
    new_msgs = result["messages"][len(input_msgs):]
    draft, file_path = _extract_saved_article(new_msgs)
    return {
        "messages": new_msgs,
        "draft_content": draft,
        "file_path": file_path,
    }


def answer_node(state: State) -> dict:
    question = state["messages"][-1].content
    summary = state.get("research_summary", "")
    draft = state.get("draft_content", "")
    response = model.invoke(
        [
            SystemMessage(
                content="你是资料助手。根据已有的调研摘要和文章内容直接回答用户问题，不要写文件。"
            ),
            HumanMessage(
                content=f"已有摘要：\n{summary}\n\n已有文章：\n{draft}\n\n用户问题：\n{question}"
            ),
        ]
    )
    return {"messages": [AIMessage(content=response.content)]}


# ---------- 主管 ----------

MAX_STEPS = 5  # 防止 writer 未写文件导致主管无限循环

INTENT_PROMPT = (
    "判断用户最新一条消息的意图，只回答一个词。\n"
    "\n"
    "三种意图：\n"
    "- 'new_task'：用户要调研并【写一篇新文章/新文档】，且主题是全新的、和之前对话无关。\n"
    "- 'followup'：用户就【之前讨论过的主题或已写的文章】提问、追问、想了解更多，"
    "例如\"是什么 / 为什么 / 区别 / 怎么理解 / 能再解释下吗\"。这类只回答，不写文件。\n"
    "- 'revise'：用户要【修改、改写、删减、扩写、润色、缩短已有的文章】。\n"
    "\n"
    "判断要点：\n"
    "- 只要最新消息是【提问】，且和之前对话的主题相关，就归为 'followup'。\n"
    "- 只有【明确是全新主题、且明确要写文章】才归为 'new_task'。\n"
    "- 提到\"改 / 删 / 扩写 / 润色 / 缩短\"等针对已有文章的操作为 'revise'。\n"
    "\n"
    "只回答 'new_task'、'followup' 或 'revise'"
)


def _classify_intent(messages) -> str:
    user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    response = model.invoke([SystemMessage(content=INTENT_PROMPT)] + user_msgs)
    intent = response.content.strip()
    if intent not in ("new_task", "followup", "revise"):
        intent = "new_task"
    return intent


def supervisor_node(state: State) -> dict:
    messages = state["messages"]
    is_new_turn = bool(messages) and isinstance(messages[-1], HumanMessage)

    if is_new_turn:
        intent = _classify_intent(messages)
        if intent == "revise" and not state.get("file_path"):
            intent = "new_task"
        if intent == "followup" and not state.get("research_summary"):
            intent = "new_task"

        if intent == "revise":
            return {"intent": intent, "next_agent": "writer"}
        if intent == "followup":
            return {"intent": intent, "next_agent": "answer"}
        return {
            "intent": intent,
            "next_agent": "researcher",
            "research_results": "",
            "research_summary": "",
            "draft_content": "",
            "file_path": "",
            "step_count": 0,
        }

    intent = state.get("intent", "new_task")
    if intent in ("followup", "revise"):
        return {"next_agent": "FINISH"}

    if state.get("file_path"):
        return {"next_agent": "FINISH"}

    step = state.get("step_count", 0) + 1
    if step > MAX_STEPS:
        return {"step_count": step, "next_agent": "FINISH"}

    if not state.get("research_results"):
        next_agent = "researcher"
    elif not state.get("research_summary"):
        next_agent = "summarizer"
    else:
        next_agent = "writer"
    return {"step_count": step, "next_agent": next_agent}


# ---------- 构建图 ----------

workflow = StateGraph(State)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("writer", writer_node)
workflow.add_node("answer", answer_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "researcher": "researcher",
        "summarizer": "summarizer",
        "writer": "writer",
        "answer": "answer",
        "FINISH": END,
    },
)

workflow.add_edge("researcher", "supervisor")
workflow.add_edge("summarizer", "supervisor")
workflow.add_edge("writer", "supervisor")
workflow.add_edge("answer", "supervisor")

graph = workflow.compile()
