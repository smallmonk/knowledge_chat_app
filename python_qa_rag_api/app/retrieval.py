import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from . import indexer


SYSTEM_PROMPT = """You are a knowledge base Q&A assistant.
Your goal is to answer questions based ONLY on the provided CONTEXT.

Rules:
1. Only answer using the provided CONTEXT.
2. Cite sources using the format [Source: filename#heading] when using information from the context.
3. If the context lacks the answer, reply exactly with "I cannot confirm from the knowledge base."
4. Do not guess or use any outside knowledge under any circumstances.
"""

_llm = None


def get_llm():
    global _llm
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set in the server environment")
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            timeout=20,
            max_retries=1,
        )
    return _llm


def build_prompt(query: str, ranked_chunks: list) -> str:
    context_parts = []
    for doc, score in ranked_chunks:
        source = doc.metadata.get("source", "unknown")
        heading = doc.metadata.get("heading", "unknown")
        context_parts.append(f"[Source: {source}#{heading}]\n{doc.page_content}")
        
    context_str = "\n\n".join(context_parts) if context_parts else "(no context)"
    
    return f"CONTEXT:\n{context_str}\n\nQUESTION:\n{query}"


def query(question: str) -> dict:
    if indexer.vectorstore is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked_chunks = indexer.search(question, k=3)
    if not ranked_chunks:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_chunks)),
    ])

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "heading": doc.metadata.get("heading", "unknown"),
            "score": round(float(score), 3),
            "content": doc.page_content[:240],
        }
        for doc, score in ranked_chunks
    ]

    answer_text = response.content
    if isinstance(answer_text, list):
        text_parts = []
        for part in answer_text:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        answer_text = "".join(text_parts)
    elif not isinstance(answer_text, str):
        answer_text = str(answer_text)

    return {
        "answer": answer_text,
        "sources": sources,
    }
