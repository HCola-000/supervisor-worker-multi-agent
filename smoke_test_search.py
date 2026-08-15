"""冒烟测试：验证 Tavily 联网搜索是否可用。

在虚拟环境终端、开着代理的情况下运行：
    python smoke_test_search.py
"""

from __future__ import annotations

import os
import sys

from dotenv import find_dotenv, load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

QUERY = "LangGraph supervisor worker 多智能体架构"


def check_env() -> str | None:
    env_path = find_dotenv()
    key = os.getenv("TAVILY_API_KEY")
    print("=" * 60)
    print("[1] 环境变量检查")
    print("=" * 60)
    print(f"  .env 位置: {env_path or '未找到'}")
    print(f"  TAVILY_API_KEY: {'已配置' if key else '未配置'}")
    return key


def test_direct_tavily(key: str) -> bool:
    from tavily import TavilyClient

    print()
    print("=" * 60)
    print("[2] 直接调用 Tavily 搜索")
    print("=" * 60)
    print(f"  查询: {QUERY}")
    client = TavilyClient(api_key=key)
    try:
        resp = client.search(query=QUERY, max_results=5)
    except Exception as exc:
        print(f"\n  ❌ 搜索失败: {type(exc).__name__}: {exc}")
        print("     排查: ①代理是否开启 ②代理是否覆盖 api.tavily.com ③key 是否有效")
        return False

    results = resp.get("results", [])
    print(f"  返回 {len(results)} 条结果:\n")
    for i, r in enumerate(results, 1):
        content = (r.get("content") or "").strip()
        print(f"  [{i}] {r.get('title', '无标题')}")
        print(f"      {r.get('url', '')}")
        print(f"      {content[:120]}{'...' if len(content) > 120 else ''}\n")
    return True


def test_real_tool() -> None:
    print("=" * 60)
    print("[3] 测试 graph.py 里的 search 工具")
    print("=" * 60)
    try:
        from src.agent.graph import search

        out = search.invoke({"query": QUERY})
        print(str(out)[:600])
    except Exception as exc:
        print("  ⚠️ 导入/调用 search 工具失败（可忽略，不影响上面结论）:")
        print(f"     {type(exc).__name__}: {exc}")


def main() -> int:
    key = check_env()
    if not key:
        print("\n请先在 D:\\Others\\Machine Learning\\.env 里添加 TAVILY_API_KEY")
        return 1
    if not test_direct_tavily(key):
        return 1
    test_real_tool()
    print()
    print("✅ 冒烟测试结束")
    return 0


if __name__ == "__main__":
    sys.exit(main())
