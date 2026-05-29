import json
import os
import re
import shutil
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

base_path: Path = Path(os.getenv("BASE_PATH", "."))

DOCS_DIR = base_path / "docs"
INDEX_DIR = base_path / ".kb" / "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# TODO: Configure chunking parameters for traditional RAG.
#
# Design decision: Balance semantic recall against context noise.
#
# Hints:
# 1. chunk_size around 500 chars is a reasonable prototype default.
# 2. chunk_overlap helps avoid cutting facts at boundaries.
# 3. separators should prefer Markdown structure before individual words.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    separators=["\n\n", "\n", ". ", " "],
)

vectorstore: FAISS | None = None
_embeddings = None
files_indexed = 0
sections_indexed = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def get_embeddings():
    global _embeddings
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the server environment")
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            request_timeout=20,
            max_retries=1,
        )
    return _embeddings


def load_markdown_sections(path: Path) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    sections = []
    
    current_heading = None
    current_level = 0
    current_content = []
    active_headings = []

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            if current_heading is not None:
                sections.append({
                    "heading": current_heading,
                    "level": current_level,
                    "active_headings": list(active_headings),
                    "content": "\n".join(current_content).strip()
                })
            
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            
            active_headings = active_headings[:level-1]
            while len(active_headings) < level - 1:
                active_headings.append("")
            active_headings.append(heading_text)
            
            current_heading = heading_text
            current_level = level
            current_content = []
        else:
            current_content.append(line)

    if current_heading is not None:
        sections.append({
            "heading": current_heading,
            "level": current_level,
            "active_headings": list(active_headings),
            "content": "\n".join(current_content).strip()
        })

    # Fallback if no headings were found in the file
    if not sections and content.strip():
        default_heading = path.stem.replace("_", " ").title()
        sections.append({
            "heading": default_heading,
            "level": 1,
            "active_headings": [default_heading],
            "content": content.strip()
        })

    documents = []
    for sec in sections:
        heading_path = " > ".join(sec["active_headings"])
        page_content = f"{heading_path}\n{sec['content']}".strip()
        slug = slugify(sec["heading"])
        source_meta = f"{path.name}#{slug}"
        
        doc = Document(
            page_content=page_content,
            metadata={
                "source": source_meta,
                "heading": sec["heading"],
                "heading_path": heading_path,
            }
        )
        documents.append(doc)

    return documents


def build_index(docs_dir: Path | None = None) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed

    if docs_dir is None:
        docs_dir = DOCS_DIR

    if not docs_dir.exists():
        vectorstore = None
        files_indexed = 0
        sections_indexed = 0
        return files_indexed, sections_indexed

    md_files = list(docs_dir.glob("*.md"))
    all_documents = []
    files_indexed = 0

    for file_path in md_files:
        try:
            docs = load_markdown_sections(file_path)
            if docs:
                all_documents.extend(docs)
                files_indexed += 1
        except Exception as exc:
            print(f"[vector_rag] Failed to load {file_path}: {exc}", flush=True)

    if not all_documents:
        vectorstore = None
        sections_indexed = 0
        save_vector_index()
        return files_indexed, sections_indexed

    chunks = splitter.split_documents(all_documents)
    sections_indexed = len(chunks)

    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    save_vector_index()

    return files_indexed, sections_indexed


def save_vector_index(index_dir: Path | None = None) -> None:
    global vectorstore, files_indexed, sections_indexed
    
    if index_dir is None:
        index_dir = INDEX_DIR

    if vectorstore is None:
        if index_dir.exists():
            shutil.rmtree(index_dir)
        return

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))

    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "files_indexed": files_indexed,
        "sections_indexed": sections_indexed
    }

    with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_vector_index(index_dir: Path | None = None) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed

    if index_dir is None:
        index_dir = INDEX_DIR

    faiss_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"
    metadata_file = index_dir / "metadata.json"

    if not (faiss_file.exists() and pkl_file.exists() and metadata_file.exists()):
        return 0, 0

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if metadata.get("embedding_model") != EMBEDDING_MODEL:
            print(f"[vector_rag] Embedding model mismatch: expected {EMBEDDING_MODEL}, found {metadata.get('embedding_model')}", flush=True)
            return 0, 0

        vectorstore = FAISS.load_local(
            str(index_dir),
            get_embeddings(),
            allow_dangerous_deserialization=True
        )

        files_indexed = metadata.get("files_indexed", 0)
        sections_indexed = metadata.get("sections_indexed", 0)
        return files_indexed, sections_indexed
    except Exception as exc:
        print(f"[vector_rag] Failed to load local vector index: {exc}", flush=True)
        return 0, 0


def search(query: str, k: int = 3) -> list[tuple[Document, float]]:
    if vectorstore is None:
        return []
    return vectorstore.similarity_search_with_score(query, k=k)
