"""
Prompt Module - ROI Cropping & Prompt Building
==============================================

Provides Region-of-Interest cropping from segmentation masks
and prompt construction for VLM inference.
"""

from llm_rag.prompt.roi_cropper import ROICropper
from llm_rag.prompt.builder import PromptBuilder

__all__ = ["ROICropper", "PromptBuilder"]
