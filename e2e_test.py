"""端到端测试：流式跑通「调研 → 总结 → 写文件」全流程。

在虚拟环境终端、开着代理的情况下运行：
    python e2e_test.py
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage

from src.agent import graph

TOPIC = "LangGraph network 多智能体架构"


def main() -> int:
    print("=" * 60)
    print("端到端测试（流式）：调研 → 总结 → 写文件")
    print("=" * 60)
    print(f"主题: {TOPIC}\n")

    input_state = {
        "messages": [
            HumanMessage(content=f"请调研「{TOPIC}」，并写一篇介绍文章。"),
        ],
    }

    current_node = None
    final_state: dict = {}

    for mode, payload in graph.stream(input_state, stream_mode=["messages", "values"]):
        if mode == "values":
            final_state = payload
        elif mode == "messages":
            msg_chunk, meta = payload
            node = meta.get("langgraph_node")
            if node and node != current_node:
                current_node = node
                print(f"\n\n----- [{node}] -----\n")
            if msg_chunk.content:
                print(msg_chunk.content, end="", flush=True)

    print("\n\n" + "=" * 60)
    print("结果")
    print("=" * 60)
    file_path = final_state.get("file_path") or ""
    print(f"  file_path: {file_path or '未生成'}")
    draft = final_state.get("draft_content") or ""
    print(f"  draft_content 长度: {len(draft)} 字")
    print()

    if file_path:
        print(f"文件已保存，路径: {file_path}")
        if os.path.exists(file_path):
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            print(f"  文件实际大小: {len(content)} 字")
            print("\n--- 文件开头 500 字预览 ---\n")
            print(content[:500])
    else:
        print("⚠️ 未检测到文件保存，流程可能未走完。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
