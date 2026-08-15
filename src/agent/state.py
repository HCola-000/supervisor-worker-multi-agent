from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import NotRequired, TypedDict


class State(TypedDict):
    """多智能体系统的共享状态"""

    messages: Annotated[Sequence[BaseMessage], add_messages]  # 对话历史
    next_agent: NotRequired[str]  # 主管决定的下一个节点名称
    intent: NotRequired[str]  # 当前轮意图: new_task / followup / revise
    step_count: NotRequired[int]  # 循环保护计数
    research_results: NotRequired[str]  # 研究员调研结果
    research_summary: NotRequired[str]  # 总结员提炼的摘要
    draft_content: NotRequired[str]  # 写手产出的正文
    file_path: NotRequired[str]  # 保存的文件绝对路径
