"""
MongoDB Atlas Vector Search for Alumni Data Retrieval

Implements the Retrieval Module (Memory) for the alumni RAG agent.
"""

import os
import certifi
from datetime import datetime
from typing import Optional, List
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class AlumniVectorStore:
    """
    MongoDB Atlas Vector Search wrapper for alumni data.
    
    Handles:
    - Document ingestion with chunking
    - Embedding generation
    - Semantic search with filters
    """
    
    def __init__(
        self,
        mongodb_uri: str = None,
        database_name: str = "alumni_db",
        collection_name: str = "alumni_vectors",
        index_name: str = "alumni_vector_index"
    ):
        """
        Initialize the vector store.
        
        Args:
            mongodb_uri: MongoDB Atlas connection string (uses MONGODB_URI env var if not provided)
            database_name: Database name
            collection_name: Collection for vector documents
            index_name: Name of the vector search index
        """
        self.mongodb_uri = mongodb_uri or os.environ.get("MONGODB_URI")
        # Disable SSL verification to bypass local network proxy/firewall inspection issues
        self.client = MongoClient(self.mongodb_uri, tlsAllowInvalidCertificates=True)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        
        # Initialize embeddings with CMU Africa AI Gateway
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(
            model="azure/text-embedding-3-small",
            openai_api_key=openai_api_key,
            base_url="https://ai-gateway.andrew.cmu.edu/"
        )
        
        # Setup chunking strategies
        self._setup_chunkers()
        
        # Initialize vector store
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name=index_name,
            text_key="content",
            embedding_key="embedding"
        )
    
    def _setup_chunkers(self):
        """Setup chunking strategies for different document types."""
        # For career narratives - preserve context with larger chunks
        self.profile_chunker = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        # For interaction logs - smaller chunks
        self.interaction_chunker = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # For policy documents - medium chunks
        self.policy_chunker = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100
        )
    
    def format_profile_text(self, profile: dict) -> str:
        """Format profile dictionary into text for embedding."""
        parts = [
            f"Alumni: {profile.get('name', 'Unknown')}",
            f"ID: {profile.get('id', 'N/A')}",
            f"Email: {profile.get('email', 'N/A')}",
            f"Graduation: {profile.get('graduation_year', 'N/A')} - {profile.get('program', 'N/A')}",
            f"Current Position: {profile.get('current_position', 'N/A')} at {profile.get('company', 'N/A')}",
            f"Location: {profile.get('location', 'N/A')}",
            f"Skills: {', '.join(profile.get('skills', []))}",
        ]
        
        if profile.get('career_history'):
            parts.append("Career History:")
            for job in profile['career_history']:
                parts.append(f"  - {job.get('title', 'N/A')} at {job.get('company', 'N/A')} ({job.get('years', 'N/A')})")
        
        return "\n".join(parts)
    
    def format_interaction_text(self, interaction: dict) -> str:
        """Format interaction log into text."""
        return f"""
Interaction with {interaction.get('alumni_name', 'Unknown')}
Date: {interaction.get('date', 'N/A')}
Type: {interaction.get('type', 'N/A')}
Summary: {interaction.get('summary', 'N/A')}
Notes: {interaction.get('notes', '')}
"""
    
    def ingest_profile(self, profile: dict) -> int:
        """
        Ingest a single alumni profile with embeddings.
        
        Args:
            profile: Alumni profile dictionary
            
        Returns:
            Number of document chunks created
        """
        text = self.format_profile_text(profile)
        chunks = self.profile_chunker.split_text(text)
        
        for i, chunk in enumerate(chunks):
            self.vector_store.add_texts(
                texts=[chunk],
                metadatas=[{
                    "alumni_id": profile.get("id"),
                    "doc_type": "profile",
                    "graduation_year": profile.get("graduation_year"),
                    "program": profile.get("program"),
                    "chunk_id": i,
                    "ingested_at": datetime.now().isoformat()
                }]
            )
        
        return len(chunks)
    
    def ingest_interaction(self, interaction: dict, alumni_id: str) -> int:
        """Ingest an interaction log."""
        text = self.format_interaction_text(interaction)
        chunks = self.interaction_chunker.split_text(text)
        
        for i, chunk in enumerate(chunks):
            self.vector_store.add_texts(
                texts=[chunk],
                metadatas=[{
                    "alumni_id": alumni_id,
                    "doc_type": "interaction",
                    "interaction_type": interaction.get("type"),
                    "chunk_id": i,
                    "ingested_at": datetime.now().isoformat()
                }]
            )
        
        return len(chunks)
    
    def ingest_policy(self, policy_text: str, policy_name: str) -> int:
        """Ingest institutional policy document."""
        chunks = self.policy_chunker.split_text(policy_text)
        
        for i, chunk in enumerate(chunks):
            self.vector_store.add_texts(
                texts=[chunk],
                metadatas=[{
                    "doc_type": "policy",
                    "policy_name": policy_name,
                    "chunk_id": i,
                    "ingested_at": datetime.now().isoformat()
                }]
            )
        
        return len(chunks)
    
    def bulk_ingest(self, profiles: List[dict]) -> int:
        """Bulk ingest multiple profiles."""
        total = 0
        for profile in profiles:
            total += self.ingest_profile(profile)
        return total
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_alumni_id: str = None,
        filter_doc_type: str = None
    ) -> list:
        """
        Search for relevant alumni documents.
        
        Args:
            query: Search query
            k: Number of results
            filter_alumni_id: Optional alumni ID filter
            filter_doc_type: Optional document type filter (profile, interaction, policy)
            
        Returns:
            List of matching documents
        """
        pre_filter = {}
        
        if filter_alumni_id:
            pre_filter["metadata.alumni_id"] = filter_alumni_id
        if filter_doc_type:
            pre_filter["metadata.doc_type"] = filter_doc_type
        
        if pre_filter:
            return self.vector_store.similarity_search(query, k=k, pre_filter=pre_filter)
        
        return self.vector_store.similarity_search(query, k=k)
    
    def search_with_score(self, query: str, k: int = 5, score_threshold: float = 0.7) -> list:
        """Search with similarity scores."""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [(doc, score) for doc, score in results if score >= score_threshold]
    
    def get_alumni_context(self, alumni_id: str, query: str) -> str:
        """Get combined context for a specific alumni."""
        # Get profile
        profile_docs = self.search(
            query=f"profile {alumni_id}",
            k=1,
            filter_alumni_id=alumni_id,
            filter_doc_type="profile"
        )
        
        # Get relevant interactions
        interaction_docs = self.search(
            query=query,
            k=3,
            filter_alumni_id=alumni_id,
            filter_doc_type="interaction"
        )
        
        # Combine context
        all_docs = profile_docs + interaction_docs
        context = "\n\n".join([doc.page_content for doc in all_docs])
        return context
