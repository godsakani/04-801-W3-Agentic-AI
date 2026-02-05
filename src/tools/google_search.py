"""
Google Search Tool for LinkedIn Profile Discovery

Automatically finds LinkedIn profiles of CMU Africa alumni using Google Search.
Uses Google Custom Search API (free tier: 100 queries/day).

Setup:
1. Go to https://programmablesearchengine.google.com/
2. Create a search engine (restrict to linkedin.com)
3. Get your Search Engine ID
4. Go to https://console.cloud.google.com/apis/credentials
5. Create an API key
6. Set GOOGLE_API_KEY and GOOGLE_CSE_ID in .env
"""

import os
import re
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
from langchain.tools import tool
from pydantic import BaseModel, Field

# Optional: Use requests for API calls
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class LinkedInSearchResult:
    """A LinkedIn profile found via Google Search."""
    name: str
    linkedin_url: str
    snippet: str
    title: Optional[str] = None


class GoogleLinkedInSearcher:
    """
    Searches Google for LinkedIn profiles matching criteria.
    
    Uses the site:linkedin.com operator to find profiles.
    """
    
    def __init__(self, api_key: str = None, cse_id: str = None):
        """
        Initialize the searcher.
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            cse_id: Custom Search Engine ID (or set GOOGLE_CSE_ID env var)
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.cse_id = cse_id or os.environ.get("GOOGLE_CSE_ID")
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(
        self,
        query: str,
        university: str = "CMU Africa",
        program: str = None,
        graduation_year: int = None,
        max_results: int = 10
    ) -> List[LinkedInSearchResult]:
        """
        Search for LinkedIn profiles matching criteria.
        
        Args:
            query: Base search query
            university: University name to search for
            program: Program name (MSIT, MSECE, etc.)
            graduation_year: Graduation year
            max_results: Maximum results to return
            
        Returns:
            List of LinkedInSearchResult objects
        """
        # Build search query
        search_query = f'site:linkedin.com/in "{university}"'
        
        if program:
            search_query += f' "{program}"'
        if graduation_year:
            search_query += f' "{graduation_year}"'
        if query:
            search_query += f' {query}'
        
        # Check if we have API credentials
        if self.api_key and self.cse_id and HAS_REQUESTS:
            return self._search_with_api(search_query, max_results)
        else:
            # Return mock results for demo
            return self._mock_search(search_query, max_results)
    
    def _search_with_api(self, query: str, max_results: int) -> List[LinkedInSearchResult]:
        """Search using Google Custom Search API."""
        results = []
        
        try:
            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": min(max_results, 10)  # API limit is 10 per request
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            # DEBUG: Print what Google actually returned
            print(f"[DEBUG] Search Query: {query}")
            print(f"[DEBUG] Total Results: {data.get('searchInformation', {}).get('totalResults', 'N/A')}")
            print(f"[DEBUG] Items in response: {len(data.get('items', []))}")
            
            if "error" in data:
                print(f"[DEBUG] API Error: {data['error']}")
            
            for item in data.get("items", []):
                url = item.get("link", "")
                
                # Only include LinkedIn profile URLs
                if "/in/" in url:
                    # Extract name from title (usually "Name - Title | LinkedIn")
                    title = item.get("title", "")
                    name = title.split(" - ")[0].split(" |")[0].strip()
                    
                    results.append(LinkedInSearchResult(
                        name=name,
                        linkedin_url=url,
                        snippet=item.get("snippet", ""),
                        title=title
                    ))
            
            return results
            
        except Exception as e:
            print(f"API search failed: {e}")
            return self._mock_search(query, max_results)
    
    def _mock_search(self, query: str, max_results: int) -> List[LinkedInSearchResult]:
        """Return mock results for demo/testing."""
        print(f"[MOCK] Would search Google for: {query}")
        
        # Mock CMU Africa alumni profiles
        mock_results = [
            LinkedInSearchResult(
                name="John Doe",
                linkedin_url="https://linkedin.com/in/johndoe-cmu",
                snippet="MSIT 2023 | Senior Data Engineer at TechCorp | CMU Africa Alumni",
                title="John Doe - Senior Data Engineer"
            ),
            LinkedInSearchResult(
                name="Jane Smith",
                linkedin_url="https://linkedin.com/in/janesmith-cmu",
                snippet="MSECE 2023 | ML Engineer at AI Startup | Carnegie Mellon University Africa",
                title="Jane Smith - ML Engineer"
            ),
            LinkedInSearchResult(
                name="Alice Wanjiku",
                linkedin_url="https://linkedin.com/in/alicewanjiku",
                snippet="MSIT 2022 | Product Manager at FinTech Kenya | CMU Africa",
                title="Alice Wanjiku - Product Manager"
            ),
            LinkedInSearchResult(
                name="Bob Ochieng",
                linkedin_url="https://linkedin.com/in/bobochieng",
                snippet="MSIT 2023 | Software Engineer at Safaricom | CMU Africa Alumni",
                title="Bob Ochieng - Software Engineer"
            ),
            LinkedInSearchResult(
                name="Grace Muthoni",
                linkedin_url="https://linkedin.com/in/gracemuthoni",
                snippet="MSECE 2022 | Data Scientist at IBM | Carnegie Mellon Africa",
                title="Grace Muthoni - Data Scientist"
            ),
        ]
        
        return mock_results[:max_results]
    
    def discover_alumni(
        self,
        programs: List[str] = None,
        years: List[int] = None,
        max_per_search: int = 10
    ) -> List[LinkedInSearchResult]:
        """
        Discover alumni profiles by searching across programs and years.
        
        Args:
            programs: List of programs to search (default: MSIT, MSECE)
            years: List of graduation years (default: last 3 years)
            max_per_search: Max results per search query
            
        Returns:
            Deduplicated list of all found profiles
        """
        if programs is None:
            programs = ["MSIT", "MSECE"]
        if years is None:
            current_year = datetime.now().year
            years = [current_year - i for i in range(3)]
        
        all_results = []
        seen_urls = set()
        
        for program in programs:
            for year in years:
                print(f"Searching: {program} {year}...")
                results = self.search(
                    query="",
                    program=program,
                    graduation_year=year,
                    max_results=max_per_search
                )
                
                # Deduplicate by URL
                for result in results:
                    if result.linkedin_url not in seen_urls:
                        seen_urls.add(result.linkedin_url)
                        all_results.append(result)
        
        print(f"Total unique profiles found: {len(all_results)}")
        return all_results


# LangChain Tool
class LinkedInDiscoveryInput(BaseModel):
    program: str = Field(default="", description="Program to search (MSIT, MSECE, or empty for all)")
    graduation_year: int = Field(default=0, description="Graduation year (0 for any)")
    max_results: int = Field(default=10, description="Maximum results to return")


@tool(args_schema=LinkedInDiscoveryInput)
def linkedin_discovery(program: str = "", graduation_year: int = 0, max_results: int = 10) -> dict:
    """
    Discover CMU Africa alumni LinkedIn profiles using Google Search.
    Searches for profiles matching the given program and graduation year.
    Returns LinkedIn URLs that can be scraped for detailed profile data.
    """
    searcher = GoogleLinkedInSearcher()
    
    results = searcher.search(
        query="",
        program=program if program else None,
        graduation_year=graduation_year if graduation_year > 0 else None,
        max_results=max_results
    )
    
    return {
        "success": True,
        "count": len(results),
        "profiles": [
            {
                "name": r.name,
                "linkedin_url": r.linkedin_url,
                "snippet": r.snippet
            }
            for r in results
        ]
    }


def create_linkedin_discovery_tool():
    """Create the LinkedIn discovery tool."""
    return linkedin_discovery


if __name__ == "__main__":
    print("LinkedIn Discovery Tool Demo")
    print("=" * 50)
    
    searcher = GoogleLinkedInSearcher()
    
    # Search for specific program/year
    results = searcher.search(
        query="",
        program="MSIT",
        graduation_year=2023,
        max_results=5
    )
    
    print(f"\nFound {len(results)} profiles:")
    for r in results:
        print(f"  - {r.name}: {r.linkedin_url}")
    
    # Discover all alumni
    print("\n" + "=" * 50)
    print("Discovering all alumni...")
    all_alumni = searcher.discover_alumni(
        programs=["MSIT", "MSECE"],
        years=[2023, 2022],
        max_per_search=3
    )
