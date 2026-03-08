"""
tools.py
────────
Lesson 2: AGENTIC SEARCH — Giving the agent live, real-world information.

Tavily is an AI-optimised search API that returns clean text summaries
(not raw HTML). It has a FREE tier: 1,000 searches/month.
Sign up at: https://app.tavily.com  →  grab your API key  →  paste in .env

Unlike a one-shot Google search, the Research Agent will call this tool
MULTIPLE TIMES with different, refined queries to build a rich picture
before writing anything.
"""

import os
from langchain_community.tools.tavily_search import TavilySearchResults

def get_search_tool(max_results: int = 3) -> TavilySearchResults:
    """
    Returns a configured Tavily search tool.

    max_results: how many web snippets to return per query.
    The agent will call this 2-3 times with different queries.
    """
    return TavilySearchResults(
        max_results=max_results,
        search_depth="advanced",      # deeper crawl for better quality
        include_answer=True,          # include Tavily's own summary answer
        include_raw_content=False,    # skip raw HTML — we want clean text
        include_images=False,
    )
