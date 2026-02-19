"""
Alumni RAG Agent — Main Agent Facade

Provides the top-level AlumniAgent class that wires up
all sub-modules: retrieval, tools, verification, memory,
and the ReAct orchestrator.
"""

import os
import json
import logging
from typing import List

from src.retrieval.mongodb_vector import AlumniVectorStore
from src.tools.linkedin import create_linkedin_tool
from src.tools.email import create_email_tool
from src.tools.survey import create_survey_tool
from src.tools.tavily_search import create_tavily_tool
from src.memory.agent_memory import PersistentMemory
from src.verification.groundedness import GroundednessScorer
from src.orchestrator import ReActAgent
from src.data.sample_alumni import SAMPLE_ALUMNI


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class AlumniAgent:
    """
    Main Alumni RAG Agent integrating all modules.
    
    Usage:
        agent = AlumniAgent()
        agent.ingest_alumni(profiles)
        result = agent.run("Find alumni in fintech and send a check-in email")
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the Alumni Agent.
        
        Args:
            config_path: Optional path to config.yaml
        """
        logger.info("Initializing Alumni RAG Agent...")
        
        # Initialize Retrieval Module
        self.retrieval = AlumniVectorStore()
        logger.info("Retrieval module initialized Successfully!")
        
        # Initialize Persistent Memory
        self.memory = PersistentMemory()
        logger.info("Persistent memory initialized Successfully!")
        
        # Initialize Tools
        self.tools = {
            "email_sender": create_email_tool(),
            "linkedin_scraper": create_linkedin_tool(),
            "survey_tool": create_survey_tool(),
            "linkedin_discovery": create_tavily_tool()
        }
        logger.info("Tools initialized: email_sender, linkedin_scraper, survey_tool, linkedin_discovery Successfully!")
        
        # Initialize Verification Module
        self.verifier = GroundednessScorer()
        logger.info("Verification module initialized Successfully!")
        
        # Create ReAct Agent with memory
        self.react_agent = ReActAgent(
            retrieval_fn=self.retrieval.search,
            tools=self.tools,
            verify_fn=self.verifier.calculate_groundedness,
            memory=self.memory
        )
        logger.info("ReAct agent initialized with role separation + persistent memory Successfully!")
        logger.info("Alumni RAG Agent ready!")
    
    def ingest_alumni(self, profiles: List[dict]) -> int:
        """
        Ingest alumni profiles into the vector store.
        
        Args:
            profiles: List of alumni profile dictionaries
            
        Returns:
            Number of document chunks created
        """
        count = self.retrieval.bulk_ingest(profiles)
        logger.info(f"Ingested {count} document chunks from {len(profiles)} profiles")
        return count
    
    def run(self, query: str, initial_observation: str = None) -> dict:
        """
        Execute a query through the ReAct loop.
        
        Args:
            query: User query
            initial_observation: Optional trigger (e.g., "LinkedIn change detected")
            
        Returns:
            {
                "response": Agent's response,
                "verification": Groundedness result,
                "trace": Execution trace for LangSmith
            }
        """
        logger.info(f"Query: {query}")
        result = self.react_agent.run(query, initial_observation)
        
        logger.info(f"Response generated with groundedness score: {result['verification'].score:.2f}")
        return result
    
    def search(self, query: str, k: int = 5) -> list:
        """Direct search of the vector store."""
        return self.retrieval.search(query, k=k)
    
    def scrape_and_ingest(self, linkedin_urls: List[str]) -> List[dict]:
        """
        Scrape LinkedIn profiles and ingest into vector store.
        
        This demonstrates the proper workflow:
        1. Use linkedin_scraper tool to get profile data
        2. Convert scraped data to alumni profile format
        3. Ingest into MongoDB vector store
        
        Args:
            linkedin_urls: List of LinkedIn profile URLs to scrape
            
        Returns:
            List of ingested profile data
        """
        logger.info(f"Scraping {len(linkedin_urls)} LinkedIn profiles...")
        
        ingested_profiles = []
        linkedin_tool = self.tools["linkedin_scraper"]
        
        for i, url in enumerate(linkedin_urls):
            # Step 1: Scrape LinkedIn using the tool
            logger.info(f"Scraping profile {i+1}/{len(linkedin_urls)}: {url}")
            scrape_result = linkedin_tool.invoke({"profile_url": url})
            
            if scrape_result.get("success"):
                profile_data = scrape_result["profile_data"]
                
                # Step 2: Convert to alumni profile format
                alumni_profile = {
                    "id": f"SCRAPED-{i+1:03d}",
                    "name": profile_data.get("name", "Unknown"),
                    "email": f"{profile_data.get('name', 'unknown').lower().replace(' ', '.')}@alumni.cmu.edu",
                    "graduation_year": 2023,  # Would be extracted in production
                    "program": "MSIT",  # Would be extracted in production
                    "linkedin_url": url,
                    "current_position": profile_data.get("current_job", ""),
                    "company": profile_data.get("company", ""),
                    "location": profile_data.get("location", ""),
                    "skills": profile_data.get("skills", []),
                    "career_history": []
                }
                
                # Step 3: Ingest into vector store
                chunks = self.retrieval.ingest_profile(alumni_profile)
                logger.info(f"Ingested {chunks} chunks for {alumni_profile['name']} Successfully!")
                
                ingested_profiles.append(alumni_profile)
                
                # Check for changes if we have previous data
                if scrape_result.get("changes"):
                    logger.info(f"  📢 Detected changes: {scrape_result['changes']}")
            else:
                logger.warning(f"Failed to scrape: {scrape_result.get('error')}")
        
        logger.info(f"Completed: Ingested {len(ingested_profiles)} profiles from LinkedIn")
        return ingested_profiles
    
    def discover_and_ingest(self, program: str = "MSIT", year: int = 2023) -> List[dict]:
        """
        Automatically discover, scrape, and ingest alumni profiles.
        
        This represents the fully automated pipeline:
        1. Search for "site:linkedin.com {program} {year}"
        2. Extract LinkedIn URLs from results
        3. Scrape each profile
        4. Ingest into MongoDB
        
        Args:
            program: Program to search for
            year: Graduation year
            
        Returns:
            List of newly ingested profiles
        """
        logger.info(f"Starting automated discovery for {program} {year}...")
        
        # Step 1: Discover profiles using Tavily Search
        discovery_tool = self.tools["linkedin_discovery"]
        search_query = f"site:linkedin.com Carnegie Mellon University Africa {program} {year}"
        
        discovery_result = discovery_tool.invoke(search_query)
        
        raw_results = discovery_result 
        if not raw_results or isinstance(raw_results, str) and "Error" in raw_results:
             logger.error(f"Discovery failed or empty: {raw_results}")
             return []
             
        profiles = raw_results 
        logger.info(f"Found {len(profiles)} search result snippets")
        
        # Step 2: Use LLM to Parse Structure
        ingested_profiles = []
        
        # Combine all search results into one context block for the LLM
        search_context = "\n---\n".join([str(p) for p in profiles])
        
        extraction_prompt = f"""You are an alumni tracking agent for CMU Africa. 
Given the raw search results below, extract structured Alumni profile data.

Search Results:
{search_context}

Return a single JSON object with a key "profiles" containing a list of objects in this exact format:
{{
    "profiles": [
        {{
            "id": "generate_unique_id",
            "name": "Full Name",
            "email": "generate_email_from_name@alumni.cmu.edu",
            "graduation_year": {year},
            "program": "{program}",
            "linkedin_url": "URL from text",
            "current_position": "Job Title",
            "company": "Company Name",
            "location": "City, Country",
            "skills": ["Python", "MongoDB", "Machine Learning"],
            "career_history": [
                {{"title": "Job Title", "company": "Company", "years": "Year-Year"}}
            ]
        }}
    ]
}}

If information is missing, use "Unknown" or empty list.
Ensure valid JSON output.
"""

        try:
            # Call the LLM (using the agent's LLM instance)
            result = self.react_agent.llm.invoke(extraction_prompt)
            content = result.content.strip()
            
            # Clean markdown JSON blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
                
            parsed_data = json.loads(content)
            extracted_profiles = parsed_data.get("profiles", [])
            
            logger.info(f"Extracted {len(extracted_profiles)} structured profiles via LLM")
            
            for profile in extracted_profiles:
                # Ingest into vector store
                chunks = self.retrieval.ingest_profile(profile)
                logger.info(f"Ingested {chunks} chunks for {profile['name']} Successfully!")
                ingested_profiles.append(profile)
                
        except Exception as e:
            logger.error(f"LLM Extraction failed: {e}")
            return []
            
        return ingested_profiles
    
    def monitor_alumni(self, alumni_list: List[dict]) -> List[dict]:
        """
        Monitor a list of alumni for LinkedIn changes and update database.
        
        This is the full closed-loop workflow:
        1. For each alumni with LinkedIn URL, scrape their profile
        2. Detect changes (job change, promotion, etc.)
        3. Update the vector store with new data
        4. Return list of alumni with changes for follow-up actions
        
        Args:
            alumni_list: List of alumni dicts with 'linkedin_url' field
            
        Returns:
            List of alumni with detected changes
        """
        logger.info(f"Monitoring {len(alumni_list)} alumni for changes...")
        
        alumni_with_changes = []
        
        for alumni in alumni_list:
            if not alumni.get("linkedin_url"):
                continue
            
            # Scrape current LinkedIn data
            scrape_result = self.tools["linkedin_scraper"].invoke({
                "profile_url": alumni["linkedin_url"]
            })
            
            if scrape_result.get("success") and scrape_result.get("changes"):
                changes = scrape_result["changes"]
                logger.info(f"📢 {alumni['name']}: Changes detected - {changes}")
                
                # Update profile in database
                updated_profile = {**alumni}
                updated_profile["current_position"] = scrape_result["profile_data"].get("current_job")
                updated_profile["company"] = scrape_result["profile_data"].get("company")
                self.retrieval.ingest_profile(updated_profile)
                
                alumni_with_changes.append({
                    "alumni": alumni,
                    "changes": changes,
                    "new_data": scrape_result["profile_data"]
                })
        
        logger.info(f"Monitoring complete: {len(alumni_with_changes)} alumni have changes")
        return alumni_with_changes


if __name__ == "__main__":
    print("Alumni RAG Agent Demo")
    print("=" * 50)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Set it to run the demo.")
    elif not os.environ.get("MONGODB_URI"):
        print("MONGODB_URI not set. Set it to run the demo.")
    else:
        agent = AlumniAgent()
        agent.ingest_alumni(SAMPLE_ALUMNI)
        
        result = agent.run("Find alumni who work in fintech")
        print(f"\nResponse: {result['response']}")
        print(f"Verification Score: {result['verification'].score:.2f}")
