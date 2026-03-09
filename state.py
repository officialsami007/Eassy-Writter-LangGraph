"""Shared state schema for the essay workflow."""

from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class EssayState(TypedDict):
    """State object passed between nodes in the LangGraph workflow."""
    
    task: str
    plan: str
    research: str
    draft: str
    critique: str
    revision_num: int
    max_revisions: int
    messages: Annotated[List[AnyMessage], add_messages]  # append-only message history
    human_feedback: str