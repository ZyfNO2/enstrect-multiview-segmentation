"""
LLM + RAG Damage Report Generation Module
==========================================

A comprehensive module for generating infrastructure damage inspection reports
using Large Language Models (LLM), Vision-Language Models (VLM),
and Retrieval-Augmented Generation (RAG) techniques.

Modules:
    - rag: PDF parsing, vector storage, and retrieval
    - vlm: Vision-language model inference and output parsing
    - prompt: ROI cropping and prompt building
    - report: Report rendering (JSON/Markdown)
    - utils: Training data collection utilities

Version: 0.1.0
Author: ENSTRECT Team
"""

__version__ = "0.1.0"
__author__ = "ENSTRECT Team"

from llm_rag.rag import PDFParser, VectorStore, RAGRetriever
from llm_rag.vlm import QwenVLInference, OutputParser
from llm_rag.prompt import ROICropper, PromptBuilder
from llm_rag.report import ReportRenderer
from llm_rag.utils import TrainingDataCollector

__all__ = [
    "PDFParser",
    "VectorStore",
    "RAGRetriever",
    "QwenVLInference",
    "OutputParser",
    "ROICropper",
    "PromptBuilder",
    "ReportRenderer",
    "TrainingDataCollector",
]
