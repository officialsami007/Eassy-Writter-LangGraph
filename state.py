"""
state.py
────────
Lesson 1: STATE — The shared TypedDict that flows through every node.
Every agent reads from it, every agent writes updates back to it.
Think of it as the essay's "working memory" — everything the system knows.
"""

from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class EssayState(TypedDict):
    """
    The central state object passed between every node in the graph.

    Lesson 1 concept: TypedDict with annotated fields.
    - task          : The original essay topic given by the user
    - plan          : Outline / plan generated before writing
    - research      : Web search results gathered by the Research Agent
    - draft         : The current draft of the essay (updated on every revision)
    - critique      : Feedback from the Critique Agent on the latest draft
    - revision_num  : Counter — how many revision cycles have run
    - max_revisions : Hard ceiling on revision cycles (prevents infinite loops)
    - messages      : Full conversation log (uses add_messages reducer so new
                      messages are APPENDED, never replaced)
    - human_feedback: Any extra feedback the human typed during HITL pause
    """
    task:           str
    plan:           str
    research:       str
    draft:          str
    critique:       str
    revision_num:   int
    max_revisions:  int
    messages:       Annotated[List[AnyMessage], add_messages]
    human_feedback: str
