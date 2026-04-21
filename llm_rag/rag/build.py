"""
CLI命令：构建RAG知识库
用法: python -m llm_rag.rag.build --pdf-dir data/standards/ --db-path data/chromadb/
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="构建RAG规范知识库")
    parser.add_argument("--pdf-dir", type=str, default="data/standards/", help="PDF规范文档目录")
    parser.add_argument("--db-path", type=str, default="data/chromadb/", help="ChromaDB存储路径")
    parser.add_argument("--sample-json", type=str, default=None, help="可选的JSON格式规范数据路径")
    args = parser.parse_args()

    from .pdf_parser import PDFParser
    from .vector_store import VectorStore

    print("=" * 60)
    print("RAG 知识库构建工具")
    print("=" * 60)

    all_docs = []

    # 从JSON加载预置规范
    sample_json = args.sample_json or str(Path(args.pdf_dir) / "sample_regulations.json")
    if Path(sample_json).exists():
        print(f"\n[1/2] 加载JSON规范数据: {sample_json}")
        json_docs = PDFParser.extract_regulations_from_json(sample_json)
        all_docs.extend(json_docs)
        print(f"       → 解析到 {len(json_docs)} 条规范片段")

    # 从PDF目录加载
    pdf_dir = Path(args.pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if pdf_files:
        print(f"\n[2/2] 解析PDF文档 ({len(pdf_files)} 个文件)")
        parser_obj = PDFParser()
        for pdf_file in pdf_files:
            try:
                docs = parser_obj.parse_and_chunk(str(pdf_file))
                all_docs.extend(docs)
                print(f"       → {pdf_file.name}: {len(docs)} 条片段")
            except Exception as e:
                print(f"       ⚠️ {pdf_file.name}: 解析失败 - {e}")

    if not all_docs:
        print("\n❌ 未找到任何规范数据！请检查 --pdf-dir 或放置 sample_regulations.json")
        return

    # 构建向量库
    print(f"\n构建向量库 (共 {len(all_docs)} 条文档)...")
    store = VectorStore(db_path=args.db_path)
    count = store.add_documents(all_docs)
    print(f"✅ 完成！已入库 {count} 条规范片段")
    print(f"   数据库路径: {args.db_path}")
    print(f"   文档总数: {store.collection_count()}")

if __name__ == "__main__":
    main()
