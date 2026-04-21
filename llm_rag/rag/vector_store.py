"""
ChromaDB向量存储模块
使用 sentence-transformers 生成嵌入，ChromaDB 持久化存储
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


class VectorStore:
    """基于 ChromaDB 的本地向量数据库，用于存储和检索规范条文嵌入"""

    DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

    def __init__(self, db_path: str, collection_name: str = "regulations",
                 embedding_model: Optional[str] = None):
        if not HAS_CHROMADB:
            raise ImportError("请安装 chromadb: pip install chromadb")
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model or self.DEFAULT_EMBEDDING_MODEL
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_model = None
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _get_embedding_model(self):
        if self.embedding_model is None:
            if not HAS_ST:
                raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        return self.embedding_model

    def add_documents(self, documents: List[Dict]) -> int:
        """添加文档到向量库，返回成功添加的数量"""
        if not documents:
            return 0
        model = self._get_embedding_model()
        ids = []
        contents = []
        metadatas = []
        for doc in documents:
            doc_id = doc.get("regulation_id", f"doc_{len(ids)}") + f"_{doc.get('chunk_id', 0)}"
            ids.append(doc_id)
            contents.append(doc["content"])
            meta = {k: v for k, v in doc.items() if k != "content" and isinstance(v, (str, int, float, bool))}
            metadatas.append(meta)
        embeddings = model.encode(contents).tolist()
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return len(ids)

    def similarity_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """查询相似度最接近的K条文档"""
        model = self._get_embedding_model()
        query_embedding = model.encode([query]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection_count())
        )
        hits = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                hits.append({
                    "content": doc,
                    "score": results["distances"][0][i] if "distances" in results else 0,
                    "metadata": results["metadatas"][0][i] if "metadatas" in results else {}
                })
        return hits

    def collection_count(self) -> int:
        return self.collection.count()

    @classmethod
    def from_json(cls, json_path: str, db_path: str) -> "VectorStore":
        """从 JSON 规范文件直接构建向量库的便捷方法"""
        from .pdf_parser import PDFParser
        store = cls(db_path=db_path)
        docs = PDFParser.extract_regulations_from_json(json_path)
        store.add_documents(docs)
        return store
