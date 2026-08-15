import pytest
from langchain_core.messages import HumanMessage

from agent import graph

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_agent_end_to_end() -> None:
    inputs = {
        "messages": [
            HumanMessage(content="请调研 LangGraph 多智能体架构并写一篇短文"),
        ],
    }
    res = await graph.ainvoke(inputs)
    assert res is not None
    assert res.get("file_path"), "writer 应保存文件并返回 file_path"
