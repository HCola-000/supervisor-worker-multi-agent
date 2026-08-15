import os

import pytest
from langchain_core.messages import HumanMessage

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_agent_end_to_end() -> None:
    if not os.getenv("API_KEY") or not os.getenv("TAVILY_API_KEY"):
        pytest.skip("未配置 API_KEY / TAVILY_API_KEY，跳过集成测试")

    # 延迟导入：没有 key 时导入 graph 会在 init_chat_model 处报错，所以要放在 skip 之后
    from agent import graph

    inputs = {
        "messages": [
            HumanMessage(content="请调研 LangGraph 多智能体架构并写一篇短文"),
        ],
    }
    res = await graph.ainvoke(inputs)
    assert res is not None
    assert res.get("file_path"), "writer 应保存文件并返回 file_path"
