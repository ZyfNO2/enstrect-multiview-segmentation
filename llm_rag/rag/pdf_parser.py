"""
PDF规范文档解析模块
支持 pdfplumber 和 pypdf2 双引擎，按章节/条款切分文本
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from pypdf2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False


class PDFParser:
    """PDF文档解析器，支持从混凝土/桥梁评定标准PDF中提取结构化文本"""

    def __init__(self, engine: str = "auto", chunk_size: int = 500, overlap: int = 50):
        self.engine = engine
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._select_engine()

    def _select_engine(self):
        if self.engine == "auto":
            if HAS_PDFPLUMBER:
                self.engine = "pdfplumber"
            elif HAS_PYPDF2:
                self.engine = "pypdf2"
            else:
                raise ImportError("请安装 pdfplumber 或 pypdf2: pip install pdfplumber pypdf2")

    def parse_pdf(self, pdf_path: str) -> List[str]:
        """解析PDF返回文本块列表（按页分割）"""
        if self.engine == "pdfplumber":
            return self._parse_with_plumber(pdf_path)
        elif self.engine == "pypdf2":
            return self._parse_with_pypdf2(pdf_path)

    def _parse_with_plumber(self, pdf_path: str) -> List[str]:
        texts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    texts.append(text.strip())
        return texts

    def _parse_with_pypdf2(self, pdf_path: str) -> List[str]:
        reader = PdfReader(pdf_path)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                texts.append(text.strip())
        return texts

    def chunk_text(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        """将长文本按固定大小分块，带重叠区域保持上下文连续性"""
        cs = chunk_size or self.chunk_size
        ov = overlap or self.overlap
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + cs, len(text))
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start += cs - ov
        return [c for c in chunks if c]

    def parse_and_chunk(self, pdf_path: str) -> List[Dict]:
        """解析PDF并分块，返回结构化文档列表"""
        raw_texts = self.parse_pdf(pdf_path)
        documents = []
        for i, text in enumerate(raw_texts):
            chunks = self.chunk_text(text)
            for j, chunk in enumerate(chunks):
                documents.append({
                    "content": chunk,
                    "source": Path(pdf_path).name,
                    "page": i + 1,
                    "chunk_id": j,
                    "metadata": {"parser": self.engine}
                })
        return documents

    @staticmethod
    def extract_regulations_from_json(json_path: str) -> List[Dict]:
        """从JSON格式规范数据加载条文（用于 sample_regulations.json 等预置数据）"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        regs = data.get("regulations", [])
        docs = []
        for reg in regs:
            content = reg.get("content", "")
            for chunk in PDFParser().chunk_text(content):
                docs.append({
                    "content": chunk,
                    "source": reg.get("source", ""),
                    "regulation_id": reg.get("id", ""),
                    "chapter": reg.get("chapter", ""),
                    "keywords": reg.get("keywords", []),
                    "damage_types": reg.get("damage_types", []),
                    "metadata": {"format": "json"}
                })
        return docs
