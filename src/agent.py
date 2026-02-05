"""
Alumni RAG Agent - Main Agent Module

Integrates the three core modules:
1. Retrieval Module - MongoDB Atlas Vector Search
2. Tool-Calling Module - LinkedIn, Email, Survey tools
3. Verification Module - Groundedness scoring

Implements a ReAct-style reasoning loop for multi-step workflows.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, List
from langchain_openai import ChatOpenAI

from src.retrieval.mongodb_vector import AlumniVectorStore
from src.tools.linkedin import create_linkedin_tool
from src.tools.email import create_email_tool
from src.tools.survey import create_survey_tool
from src.tools.tavily_search import create_tavily_tool
# from src.tools.duck_search import create_linkedin_discovery_tool
from src.verification.groundedness import GroundednessScorer, GroundednessResult, handle_verification


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Tracks agent state across loop iterations."""
    query: str
    context: str = ""
    observations: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    iteration: int = 0
    current_alumni_id: Optional[str] = None


class ReActAgent:
    """
    ReAct-style agent for alumni tracking.
    
    Loop: OBSERVE → REASON → DECIDE → ACT → UPDATE → REPEAT
    
    Decision Point: Agent chooses between:
    - RETRIEVAL_MODULE: Need context from alumni database
    - TOOL_MODULE: Need external action (email, scrape, survey)
    - FINAL_ANSWER: Ready to respond
    """
    
    def __init__(
        self,
        retrieval_fn: Callable,
        tools: dict,
        verify_fn: Callable,
        max_iterations: int = 5
    ):
        self.retrieval_fn = retrieval_fn
        self.tools = tools
        self.verify_fn = verify_fn
        self.max_iterations = max_iterations
        self.llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0, base_url='https://ai-gateway.andrew.cmu.edu/', openai_api_key=os.environ.get("OPENAI_API_KEY"))
    
    def _log(self, phase: str, content: str):
        """Log to both console and file."""
        logger.info(f"[{phase}] {content[:100]}...")
    
    def _log_decision_point(self, decision: str, options: list, selected: str):
        """Log a decision point for the implementation trace."""
        logger.info(f"DECISION POINT: Options={options}, Selected={selected}")
    
    def run(self, query: str, initial_observation: str = None) -> dict:
        """
        Execute the ReAct loop.
        
        Args:
            query: User query or trigger
            initial_observation: Optional initial observation (e.g., LinkedIn change)
            
        Returns:
            {
                "response": final response,
                "verification": groundedness result,
                "trace": list of steps
            }
        """
        state = AgentState(query=query)
        
        if initial_observation:
            state.observations.append(initial_observation)
        
        trace = []
        
        while state.iteration < self.max_iterations:
            state.iteration += 1
            self._log("ITERATION", f"Starting iteration {state.iteration}")
            
            # ========== REASON ==========
            thought = self._reason(state)
            trace.append({"phase": "REASON", "thought": thought, "iteration": state.iteration})
            self._log("REASON", thought)
            
            # ========== DECIDE ==========
            decision = self._parse_decision(thought)
            options = ["RETRIEVE", "email_sender", "linkedin_scraper", "survey_tool", "FINAL_ANSWER"]
            self._log_decision_point(decision, options, decision)
            trace.append({"phase": "DECIDE", "decision": decision, "iteration": state.iteration})
            
            # ========== FINAL ANSWER ==========
            if decision == "FINAL_ANSWER":
                self._log("DECIDE", "Ready to generate final answer")
                break
            
            # ========== ACT: RETRIEVAL MODULE ==========
            if decision == "RETRIEVE":
                self._log("ACT", "Executing RETRIEVAL_MODULE")
                retrieval_query = self._extract_retrieval_query(thought)
                docs = self.retrieval_fn(retrieval_query or state.query)
                
                # Update context
                new_context = "\n\n".join([doc.page_content for doc in docs])
                state.context += "\n\n" + new_context
                state.observations.append(f"Retrieved {len(docs)} documents")
                state.actions.append("RETRIEVAL_MODULE")
                
                trace.append({
                    "phase": "ACT",
                    "module": "RETRIEVAL_MODULE",
                    "result": f"Retrieved {len(docs)} documents",
                    "iteration": state.iteration
                })
            
            # ========== ACT: TOOL MODULE ==========
            elif decision in self.tools:
                self._log("ACT", f"Executing TOOL_MODULE: {decision}")
                tool_fn = self.tools[decision]
                
                # Extract parameters and execute
                params = self._extract_tool_params(thought, decision, state.context)
                
                try:
                    result = tool_fn.invoke(params) if hasattr(tool_fn, 'invoke') else {"mock": True}
                    state.observations.append(f"Executed {decision}: {result}")
                    state.actions.append(f"TOOL_MODULE:{decision}")
                    
                    trace.append({
                        "phase": "ACT",
                        "module": "TOOL_MODULE",
                        "tool": decision,
                        "result": str(result)[:100],
                        "iteration": state.iteration
                    })
                except Exception as e:
                    state.observations.append(f"Tool {decision} failed: {e}")
                    trace.append({
                        "phase": "ACT",
                        "module": "TOOL_MODULE",
                        "tool": decision,
                        "error": str(e),
                        "iteration": state.iteration
                    })
        
        # ========== GENERATE RESPONSE ==========
        response = self._generate_response(state)
        self._log("RESPONSE", response)
        
        # ========== VERIFY ==========
        self._log("VERIFY", "Calculating groundedness score")
        sources = [state.context] if state.context else []
        verification = self.verify_fn(response, sources)
        
        trace.append({
            "phase": "VERIFY",
            "score": verification.score,
            "confidence": verification.confidence,
            "recommendation": verification.recommendation
        })
        
        # Handle verification result
        final_response = handle_verification(verification, response)
        
        return {
            "response": final_response,
            "verification": verification,
            "trace": trace
        }
    
    def _reason(self, state: AgentState) -> str:
        """Generate reasoning about what to do next."""
        prompt = f"""You are an alumni tracking agent for CMU Africa. 
Given the query and context, decide what action to take next.

Query: {state.query}

Current context: {state.context[:1500] if state.context else 'No context retrieved yet'}

Previous observations: {state.observations}
Previous actions: {state.actions}

Available actions:
1. RETRIEVE - Search the alumni database for relevant information
2. email_sender - Send a personalized email to an alumni
3. linkedin_scraper - Check an alumni's LinkedIn profile for updates
4. survey_tool - Send a survey to collect feedback
5. FINAL_ANSWER - I have enough information to respond

Think step by step:
1. What information do I need?
2. Do I have enough context to answer?
3. What action should I take?

End your response with: DECISION: <action_name>"""

        result = self.llm.invoke(prompt)
        return result.content
    
    def _parse_decision(self, thought: str) -> str:
        """Parse decision from LLM reasoning."""
        if "DECISION:" in thought:
            parts = thought.split("DECISION:")
            if len(parts) > 1:
                decision = parts[1].strip().split()[0].upper()
                # Normalize tool names
                tool_map = {
                    "EMAIL_SENDER": "email_sender",
                    "LINKEDIN_SCRAPER": "linkedin_scraper",
                    "SURVEY_TOOL": "survey_tool"
                }
                return tool_map.get(decision, decision)
        return "FINAL_ANSWER"
    
    def _extract_retrieval_query(self, thought: str) -> str:
        """Extract what to search for from reasoning."""
        # Simple extraction - in production, use more sophisticated parsing
        return None  # Use original query
    
    def _extract_tool_params(self, thought: str, tool_name: str, context: str = "") -> dict:
        """Extract tool parameters from reasoning using LLM."""
        import json
        prompt = f"""Extract the parameters for the tool '{tool_name}' based on the reasoning and context below.
Return ONLY a valid JSON object.

Reasoning:
{thought}

Context (Database Records):
{context[:2000] if context else 'No context'}

Tool Schema:

Tool Schema:
- email_sender: {{ "recipient_email": "string", "template": "general_check_in | congratulations_promotion | congratulations_new_job | offer_support | survey_request", "personalization": {{ "name": "string" }} }}
- linkedin_scraper: {{ "profile_url": "string" }}
- survey_tool: {{ "questions": ["string"], "topic": "string" }}

JSON Output:"""
        
        try:
            result = self.llm.invoke(prompt)
            content = result.content.strip()
            
            # Remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
                
            params = json.loads(content)
            return params
        except Exception as e:
            print(f"Parameter extraction failed: {e}")
            # Fallback for demo stability if parsing fails
            # if tool_name == "email_sender":
            #     return {
            #         "recipient_email": "alumni@example.com",
            #         "template": "general_check_in",
            #         "personalization": {"name": "Alumni"}
            #     }
            return {}
    
    def _generate_response(self, state: AgentState) -> str:
        """Generate final response from accumulated context."""
        prompt = f"""Based on the following context and actions, generate a helpful response to the user's query.

Query: {state.query}

Retrieved Context:
{state.context[:2000] if state.context else 'No specific context available'}

Actions Taken: {state.actions}
Observations: {state.observations}

Generate a clear, accurate, and helpful response. Only include information that is supported by the context."""

        result = self.llm.invoke(prompt)
        return result.content


class AlumniAgent:
    """
    Main Alumni RAG Agent integrating all three modules.
    
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
        logger.info("✓ Retrieval module initialized")
        
        # Initialize Tools
        self.tools = {
            "email_sender": create_email_tool(),
            "linkedin_scraper": create_linkedin_tool(),
            "survey_tool": create_survey_tool(),
            "linkedin_discovery": create_tavily_tool()
        }
        logger.info("✓ Tools initialized: email_sender, linkedin_scraper, survey_tool, linkedin_discovery")
        
        # Initialize Verification Module
        self.verifier = GroundednessScorer()
        logger.info("✓ Verification module initialized")
        
        # Create ReAct Agent
        self.react_agent = ReActAgent(
            retrieval_fn=self.retrieval.search,
            tools=self.tools,
            verify_fn=self.verifier.calculate_groundedness
        )
        logger.info("✓ ReAct agent initialized")
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
                logger.info(f"  ✓ Ingested {chunks} chunks for {alumni_profile['name']}")
                
                ingested_profiles.append(alumni_profile)
                
                # Check for changes if we have previous data
                if scrape_result.get("changes"):
                    logger.info(f"  📢 Detected changes: {scrape_result['changes']}")
            else:
                logger.warning(f"  ✗ Failed to scrape: {scrape_result.get('error')}")
        
        logger.info(f"Completed: Ingested {len(ingested_profiles)} profiles from LinkedIn")
        return ingested_profiles
    
    def discover_and_ingest(self, program: str = "MSIT", year: int = 2023) -> List[dict]:
        """
        Automatically discover, scrape, and ingest alumni profiles.
        
        This represents the fully automated pipeline:
        1. Search Google for "site:linkedin.com {program} {year}"
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
        
        # Step 1: Discover profiles using Google Search
        discovery_tool = self.tools["linkedin_discovery"]
        # Construct a search query for Tavily
        # e.g. "site:linkedin.com Carnegie Mellon University Africa MSIT 2023"
        search_query = f"site:linkedin.com Carnegie Mellon University Africa {program} {year}"
        
        discovery_result = discovery_tool.invoke(search_query)
        
        # The Tavily tool returns a list of formatted string snippets, not a dictionary with "profiles"
        # So discovery_tool.invoke returns List[str] directly.
        
        raw_results = discovery_result 
        if not raw_results or isinstance(raw_results, str) and "Error" in raw_results:
             logger.error(f"Discovery failed or empty: {raw_results}")
             return []
             
        # "profiles" in this variable name implies user personas, but here it's just text snippets
        profiles = raw_results 
        logger.info(f"Found {len(profiles)} search result snippets")
        
        # Step 2: Use LLM to Parse Structure (replacing the old scrape loop)
        # We process the text directly because Tavily gives us the content.
            
        # Re-use our existing scrape pipeline
        # return self.scrape_and_ingest(linkedin_urls)
        
        # NOTE: Since Tavily gives us raw content directly, we can skip the specialized scraper 
        # for now and use LLM to extract profile data directly from the search snippets.
        # This is more robust than scraping URLs which blocks bots.
        
        ingested_profiles = []
        import json
        
        # Combine all search results into one context block for the LLM
        # profiles here contains string snippets from our tavily tool wrapper
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
                logger.info(f"  ✓ Ingested {chunks} chunks for {profile['name']}")
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


# Sample alumni data for testing
SAMPLE_ALUMNI = [
    {
        "id": "A2023-001",
        "name": "John Doe",
        "email": "nyonggodwill11@gmail.com",
        "graduation_year": 2023,
        "program": "MSIT",
        "linkedin_url": "https://linkedin.com/in/johndoe",
        "current_position": "Senior Data Engineer",
        "company": "TechCorp",
        "location": "Nairobi, Kenya",
        "skills": ["Python", "MongoDB", "Machine Learning"],
        "career_history": [
            {"title": "Junior Developer", "company": "StartupXYZ", "years": "2021-2023"},
            {"title": "Senior Data Engineer", "company": "TechCorp", "years": "2023-present"}
        ]
    },
    {
        "id": "A2023-002",
        "name": "Jane Smith",
        "email": "nyonggodwill95@gmail.com",
        "graduation_year": 2023,
        "program": "MSECE",
        "linkedin_url": "https://linkedin.com/in/janesmith",
        "current_position": "ML Engineer",
        "company": "AI Startup",
        "location": "Kigali, Rwanda",
        "skills": ["TensorFlow", "PyTorch", "Computer Vision"],
        "career_history": [
            {"title": "Research Assistant", "company": "CMU Africa", "years": "2022-2023"},
            {"title": "ML Engineer", "company": "AI Startup", "years": "2023-present"}
        ]
    },
    {
        "id": "A2022-015",
        "name": "Alice Wanjiku",
        "email": "nyonggodwill@gmail.com",
        "graduation_year": 2022,
        "program": "MSIT",
        "linkedin_url": "https://linkedin.com/in/alicew",
        "current_position": "Product Manager",
        "company": "FinTech Kenya",
        "location": "Nairobi, Kenya",
        "skills": ["Product Strategy", "Agile", "Data Analysis"],
        "career_history": [
            {"title": "Software Engineer", "company": "Safaricom", "years": "2022-2024"},
            {"title": "Product Manager", "company": "FinTech Kenya", "years": "2024-present"}
        ]
    }
]


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
