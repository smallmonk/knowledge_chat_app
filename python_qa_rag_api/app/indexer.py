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


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed

    # TODO: Build a FAISS vector index from docs/*.md.
    #
    # Hints:
    # 1. Load all Markdown files from docs_dir.
    # 2. Convert each heading section to a Document.
    # 3. Split documents into chunks with splitter.split_documents().
    # 4. Create FAISS.from_documents(chunks, get_embeddings()).
    # 5. Save the FAISS index to .kb/faiss_index/.
    # 6. Return (files_indexed, chunks_indexed).
    vectorstore = None
    files_indexed = 0
    sections_indexed = 0
    return files_indexed, sections_indexed


def save_vector_index(index_dir: Path = INDEX_DIR) -> None:
    # TODO: Persist the FAISS index so restart does not require re-embedding.
    #
    # Hints:
    # 1. Return early if vectorstore is None.
    # 2. Clear stale persisted files with shutil.rmtree(...) if the new index is empty.
    # 3. Use vectorstore.save_local(str(index_dir)).
    # 4. Write metadata.json with embedding_model, files_indexed, and sections_indexed.
    # 5. json.dumps(..., indent=2) makes the metadata easy to inspect.
    pass


def load_vector_index(index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    # TODO: Load .kb/faiss_index/ on server startup if it exists.
    #
    # Hints:
    # 1. Check for index.faiss and index.pkl.
    # 2. Read metadata.json and verify embedding_model still matches.
    # 3. Use FAISS.load_local(..., allow_dangerous_deserialization=True).
    # 4. Only use dangerous deserialization for indexes created by this local app.
    return 0, 0


def search(query: str, k: int = 3) -> list[tuple[Document, float]]:
    if vectorstore is None:
        return []
    return vectorstore.similarity_search_with_score(query, k=k)
