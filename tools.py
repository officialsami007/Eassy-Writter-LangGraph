"""Search tool configuration."""


import os
from langchain_comamunity.tools.tavily_search import TavilySearchResults

def get_search_tool(max_results: int = 3) -> TavilySearchResults:
    """Return a configured Tavily search tool."""

    return TavilySearchResults(
        max_results=max_results,
        search_depth="advanced",      # deeper crawl for better quality
        include_answer=True,          # include Tavily's own summary answer
        include_raw_content=False,    # skip raw HTML — we want clean text
        include_images=False,
    )
