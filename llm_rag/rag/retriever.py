"""
RAG检索增强生成模块
统一检索接口，支持关键词+语义混合检索
"""

from typing import List, Dict, Optional
from .vector_store import VectorStore
from .pdf_parser import PDFParser


class RAGRetriever:
    """RAG检索器，封装向量库检索逻辑，提供面向损伤检测的高级接口"""

    DAMAGE_TYPE_KEYWORDS = {
        "crack": ["裂缝", "裂纹", "开裂", "宽度", "长度", "贯穿"],
        "spalling": ["剥落", "脱落", "破损", "露石", "蜂窝"],
        "corrosion": ["腐蚀", "锈蚀", "钢筋锈", "锈斑", "氧化"],
        "efflorescence": ["泛白", "风化", "盐析", "白霜"],
        "vegetation": ["植被", "苔藓", "杂草", "植物生长"],
        "exposed_rebar": ["露筋", "钢筋外露", "保护层不足"]
    }

    SEVERITY_KEYWORDS = {
        "minor": ["轻微", "一般", "一级", "二级", "正常"],
        "moderate": ["中等", "较严重", "三级"],
        "severe": ["严重", "四级", "危险"],
        "critical": ["危桥", "五级", "紧急"]
    }

    def __init__(self, vector_store: VectorStore, top_k: int = 3):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        主检索接口：根据自然语言查询返回相关规范条文
        
        Args:
            query: 查询文本，如"混凝土表面有0.3mm宽裂缝如何评定"
            top_k: 返回条数
            
        Returns:
            包含 content, score, metadata 的字典列表
        """
        k = top_k or self.top_k
        return self.vector_store.similarity_search(query, k)

    def retrieve_by_damage_type(self, damage_type: str, severity_hint: Optional[str] = None) -> List[Dict]:
        """
        根据损伤类型检索相关规范
        
        Args:
            damage_type: 损伤类型英文标识 (crack/spalling/corrosion等)
            severity_hint: 可选的严重程度提示词
        """
        keywords = self.DAMAGE_TYPE_KEYWORDS.get(damage_type, [damage_type])
        query = " ".join(keywords[:3])
        if severity_hint:
            sev_kw = self.SEVERITY_KEYWORDS.get(severity_hint, [severity_hint])
            query += " " + " ".join(sev_kw[:2])
        return self.retrieve(query)

    def build_context(self, retrieval_results: List[Dict]) -> str:
        """将检索结果拼接为可注入Prompt的上下文文本"""
        if not retrieval_results:
            return ""
        lines = ["【参考规范条文】"]
        for i, hit in enumerate(retrieval_results, 1):
            meta = hit.get("metadata", {})
            source = meta.get("source", "未知来源")
            reg_id = meta.get("regulation_id", "")
            chapter = meta.get("chapter", "")
            header = f"条目{i}：{source}"
            if reg_id:
                header += f" ({reg_id})"
            if chapter:
                header += f" - {chapter}"
            lines.append(header)
            lines.append(hit["content"])
            lines.append("")
        return "\n".join(lines)

    @classmethod
    def from_sample_data(cls, json_path: str, db_path: str, **kwargs) -> "RAGRetriever":
        """从示例规范JSON快速构建RAG检索器的便捷工厂方法"""
        store = VectorStore.from_json(json_path, db_path)
        return cls(store, **kwargs)
