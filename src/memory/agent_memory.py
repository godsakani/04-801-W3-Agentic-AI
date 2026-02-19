"""
Persistent Memory Module — Cross-Session Memory for Alumni RAG Agent

Provides long-term memory backed by MongoDB for:
- Session history (queries, responses, tools used, metrics)
- User preferences
- Task history lookup

Memory Policies:
- Write: After every agent.run() completes
- Read: At start of each run() — inject prior session context
- Prune: Summarize sessions older than 30 days, delete raw data
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from pymongo import MongoClient, DESCENDING

logger = logging.getLogger(__name__)


class PersistentMemory:
    """
    Cross-session persistent memory backed by MongoDB.
    
    Uses a dedicated 'agent_memory' collection in the same
    alumni_db database to store session history, preferences,
    and task outputs.
    
    Memory Types:
    - Session records: query, response summary, tools used, metrics
    - User preferences: key-value pairs per user
    - Summarized history: compressed older sessions
    """
    
    def __init__(
        self,
        mongodb_uri: str = None,
        database_name: str = "alumni_db",
        collection_name: str = "agent_memory"
    ):
        """
        Initialize persistent memory.
        
        Args:
            mongodb_uri: MongoDB Atlas connection string (uses MONGODB_URI env var)
            database_name: Database name (reuses alumni_db)
            collection_name: Collection for memory records
        """
        self.mongodb_uri = mongodb_uri or os.environ.get("MONGODB_URI")
        self.client = MongoClient(self.mongodb_uri, tlsAllowInvalidCertificates=True)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        
        # Ensure indexes for efficient queries
        self.collection.create_index([("timestamp", DESCENDING)])
        self.collection.create_index([("type", 1)])
        self.collection.create_index([("session_id", 1)])
        
        logger.info(f"PersistentMemory initialized: {database_name}.{collection_name}")
    
    # ================================================================
    # WRITE POLICY: Save after every agent.run()
    # ================================================================
    
    def save_session(
        self,
        session_id: str,
        query: str,
        response: str,
        tools_used: List[str],
        metrics: dict,
        trace_summary: str = ""
    ) -> str:
        """
        Save a completed session to persistent memory.
        
        Called after every agent.run() completes — this is the core
        write policy. Captures everything needed to reconstruct context.
        
        Args:
            session_id: Unique session identifier
            query: User's original query
            response: Agent's final response (truncated for storage)
            tools_used: List of tools that were invoked
            metrics: Dict with groundedness_score, iterations, etc.
            trace_summary: Brief summary of the execution trace
            
        Returns:
            MongoDB document ID as string
        """
        record = {
            "type": "session",
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
            "query": query,
            "response_summary": response[:500],  # Truncate for storage efficiency
            "tools_used": tools_used,
            "metrics": metrics,
            "trace_summary": trace_summary,
        }
        
        result = self.collection.insert_one(record)
        logger.info(f"[MEMORY WRITE] Session {session_id} saved: query='{query[:50]}...', "
                    f"tools={tools_used}, groundedness={metrics.get('groundedness_score', 'N/A')}")
        return str(result.inserted_id)
    
    def save_user_preference(self, user_id: str, key: str, value: str) -> str:
        """
        Save or update a user preference.
        
        Called when the user explicitly states a preference
        (e.g., "I prefer detailed responses" or "Always search fintech alumni first").
        
        Args:
            user_id: User identifier
            key: Preference key (e.g., "response_style", "default_filter")
            value: Preference value
            
        Returns:
            MongoDB document ID as string
        """
        result = self.collection.update_one(
            {"type": "preference", "user_id": user_id, "key": key},
            {
                "$set": {
                    "value": value,
                    "updated_at": datetime.utcnow()
                },
                "$setOnInsert": {
                    "created_at": datetime.utcnow()
                }
            },
            upsert=True
        )
        
        logger.info(f"[MEMORY WRITE] Preference saved: user={user_id}, {key}={value}")
        return str(result.upserted_id or "updated")
    
    # ================================================================
    # READ POLICY: Load at start of each run()
    # ================================================================
    
    def get_recent_sessions(self, limit: int = 5) -> List[dict]:
        """
        Retrieve recent session records for context injection.
        
        Called at the start of each agent.run() to provide
        cross-session continuity. Returns the most recent sessions
        so the agent can reference prior interactions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session dicts (most recent first)
        """
        cursor = self.collection.find(
            {"type": "session"},
            {"_id": 0, "query": 1, "response_summary": 1, 
             "tools_used": 1, "metrics": 1, "timestamp": 1, "session_id": 1}
        ).sort("timestamp", DESCENDING).limit(limit)
        
        sessions = list(cursor)
        logger.info(f"[MEMORY READ] Loaded {len(sessions)} recent sessions")
        return sessions
    
    def get_user_preferences(self, user_id: str) -> dict:
        """
        Retrieve all preferences for a user.
        
        Called at start of run() to personalize behavior.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict of {key: value} preferences
        """
        cursor = self.collection.find(
            {"type": "preference", "user_id": user_id},
            {"_id": 0, "key": 1, "value": 1}
        )
        
        prefs = {doc["key"]: doc["value"] for doc in cursor}
        logger.info(f"[MEMORY READ] Loaded {len(prefs)} preferences for user={user_id}")
        return prefs
    
    def get_task_history(self, query: str, limit: int = 3) -> List[dict]:
        """
        Find similar past queries to inform the current run.
        
        Uses text matching to find sessions with related queries.
        Helps the agent avoid repeating work or build on prior results.
        
        Args:
            query: Current query to search for similar past tasks
            limit: Maximum results to return
            
        Returns:
            List of matching session records
        """
        # Simple keyword search — extract key terms from query
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        
        if not keywords:
            return []
        
        # Search for sessions containing any of the keywords
        regex_pattern = "|".join(keywords[:5])  # Limit to 5 keywords
        cursor = self.collection.find(
            {
                "type": "session",
                "query": {"$regex": regex_pattern, "$options": "i"}
            },
            {"_id": 0, "query": 1, "response_summary": 1,
             "tools_used": 1, "timestamp": 1, "session_id": 1}
        ).sort("timestamp", DESCENDING).limit(limit)
        
        matches = list(cursor)
        logger.info(f"[MEMORY READ] Found {len(matches)} similar past tasks for: '{query[:50]}'")
        return matches
    
    # ================================================================
    # PRUNING POLICY: Summarize and clean old sessions
    # ================================================================
    
    def prune_old_sessions(self, max_age_days: int = 30) -> dict:
        """
        Prune sessions older than max_age_days.
        
        Strategy:
        1. Find all sessions older than threshold
        2. Create a summary record capturing key info
        3. Delete the raw session records
        
        This prevents memory from growing infinitely while preserving
        the essence of past interactions.
        
        Args:
            max_age_days: Maximum age in days before pruning
            
        Returns:
            Dict with pruning stats
        """
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        
        # Find old sessions
        old_sessions = list(self.collection.find(
            {"type": "session", "timestamp": {"$lt": cutoff}}
        ))
        
        if not old_sessions:
            logger.info("[MEMORY PRUNE] No sessions to prune")
            return {"pruned": 0, "summary_created": False}
        
        # Create summary of old sessions
        summary_text = self._summarize_sessions(old_sessions)
        
        # Store the summary
        self.collection.insert_one({
            "type": "summary",
            "timestamp": datetime.utcnow(),
            "period_start": min(s["timestamp"] for s in old_sessions),
            "period_end": max(s["timestamp"] for s in old_sessions),
            "session_count": len(old_sessions),
            "summary": summary_text
        })
        
        # Delete old raw sessions
        result = self.collection.delete_many(
            {"type": "session", "timestamp": {"$lt": cutoff}}
        )
        
        logger.info(f"[MEMORY PRUNE] Pruned {result.deleted_count} old sessions, "
                    f"created summary covering {len(old_sessions)} sessions")
        
        return {
            "pruned": result.deleted_count,
            "summary_created": True,
            "summary": summary_text[:200]
        }
    
    def _summarize_sessions(self, sessions: list) -> str:
        """
        Create a text summary of multiple sessions.
        
        Uses a simple template approach — for production, this could
        use an LLM to generate a more natural summary.
        
        Args:
            sessions: List of session records to summarize
            
        Returns:
            Summary text
        """
        queries = [s.get("query", "Unknown") for s in sessions]
        tools_used = set()
        total_groundedness = 0
        count = 0
        
        for s in sessions:
            tools_used.update(s.get("tools_used", []))
            metrics = s.get("metrics", {})
            if "groundedness_score" in metrics:
                total_groundedness += metrics["groundedness_score"]
                count += 1
        
        avg_groundedness = total_groundedness / count if count > 0 else 0
        
        summary = (
            f"Summary of {len(sessions)} past sessions:\n"
            f"- Queries covered: {'; '.join(queries[:10])}\n"
            f"- Tools used: {', '.join(tools_used) if tools_used else 'None'}\n"
            f"- Average groundedness: {avg_groundedness:.2f}\n"
            f"- Period: {sessions[-1].get('timestamp', 'N/A')} to {sessions[0].get('timestamp', 'N/A')}"
        )
        
        return summary
    
    # ================================================================
    # Helper: Format memory context for injection into agent state
    # ================================================================
    
    def format_memory_context(self, sessions: List[dict], preferences: dict = None) -> str:
        """
        Format memory data into a context string for injection into AgentState.
        
        Args:
            sessions: Recent session records
            preferences: User preference dict
            
        Returns:
            Formatted context string for the LLM
        """
        if not sessions and not preferences:
            return ""
        
        parts = []
        
        if preferences:
            prefs_str = ", ".join([f"{k}: {v}" for k, v in preferences.items()])
            parts.append(f"User Preferences: {prefs_str}")
        
        if sessions:
            parts.append("Recent Session History:")
            for i, session in enumerate(sessions[:3], 1):
                ts = session.get("timestamp", "")
                if isinstance(ts, datetime):
                    ts = ts.strftime("%Y-%m-%d %H:%M")
                parts.append(
                    f"  [{i}] Query: {session.get('query', 'N/A')[:100]} | "
                    f"Tools: {', '.join(session.get('tools_used', []))} | "
                    f"Time: {ts}"
                )
        
        return "\n".join(parts)
