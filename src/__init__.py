"""
Alumni RAG Agent - Main Package

A RAG-enabled agentic system for CMU Africa alumni tracking and support.
"""

from src.agent import AlumniAgent
from src.data.sample_alumni import SAMPLE_ALUMNI

__version__ = "2.0.0"
__all__ = ["AlumniAgent", "SAMPLE_ALUMNI"]
