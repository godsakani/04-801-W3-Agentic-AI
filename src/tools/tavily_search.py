"""
Tavily Search Tool for Alumni Discovery
Uses Tavily API to find alumni profiles, replacing Google Custom Search.
"""

import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from typing import List

# Initialize the base Tavily tool
tavily_wrapper = TavilySearchResults(
    max_results=10,
    include_answer=True,
    include_raw_content=True,
    include_images=False,  # Images likely not needed for text profiling
    include_domains=["linkedin.com/in"],
    # tavily_api_key is read from os.environ["TAVILY_API_KEY"] automatically
)

@tool
def tavily_discovery(query: str) -> List[str]:
    """
    Search for alumni LinkedIn profiles using Tavily.
    Useful for discovering new alumni or verifying current roles.
    Input should be a search query like 'John Doe CMU Africa LinkedIn'.
    """
    try:
        results = tavily_wrapper.invoke(query)
        
        # Extract just values if results is a list of dicts (standard Tavily output)
        # We want to return a list of readable snippets or URLs
        formatted_results = []
        if isinstance(results, list):
            for res in results:
                url = res.get("url", "")
                content = res.get("content", "")
                formatted_results.append(f"URL: {url}\nContent: {content}")
        else:
            return [str(results)]
            
        return formatted_results
        
    except Exception as e:
        return [f"Error searching Tavily: {str(e)}"]

def create_tavily_tool():
    """Factory function to get the tool (consistent with other tools)."""
    # Ensure API key is present or warn
    if not os.environ.get("TAVILY_API_KEY"):
        print("WARNING: TAVILY_API_KEY not found in environment.")
    return tavily_discovery
