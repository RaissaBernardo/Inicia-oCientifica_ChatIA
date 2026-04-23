"""
main.py - Servidor FastAPI com RAG para o chatbot da Carda TC 15.
Rodar:  python -m uvicorn main:app --reload  (porta padrão: 8000)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Configurações ──────────────────────────────────────────────
CHROMA_DIR   = Path(__file__).parent / "chroma_db"
EMBED_MODEL  = "nomic-embed-text"
LLM_MODEL    = "llama3"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
TOP_K        = 5
# ──────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """Você é um assistente técnico especializado na máquina Carda TC 15 (2017).
Responda em português brasileiro, de forma clara e objetiva.
Use APENAS as informações fornecidas no contexto abaixo.
Se a informação não estiver no contexto, diga: "Não encontrei essa informação na documentação da Carda TC 15."

Contexto extraído do manual:
{context}

Pergunta do operador: {question}

Resposta técnica:"""

app = FastAPI(title="Chatbot Carda TC 15", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain = None
retriever = None

@app.on_event("startup")
async def startup():
    global rag_chain, retriever

    if not CHROMA_DIR.exists():
        print("[AVISO] Banco vetorial não encontrado. Rode 'python ingest.py' primeiro.")
        return

    print("[startup] Carregando ChromaDB...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})

    print("[startup] Conectando ao Ollama LLM...")
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.1, num_predict=512)

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("[startup] ✅  RAG pronto!")

class ChatRequest(BaseModel):
    question: str

class SourceInfo(BaseModel):
    page: int
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Pergunta vazia.")

    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Sistema RAG não inicializado. Rode 'python ingest.py' e reinicie o servidor."
        )

    answer = rag_chain.invoke(req.question)

    docs = retriever.invoke(req.question)
    seen, sources = set(), []
    for doc in docs:
        page = doc.metadata.get("page", 0) + 1
        if page not in seen:
            seen.add(page)
            sources.append(SourceInfo(
                page=page,
                snippet=doc.page_content[:180].replace("\n", " ") + "..."
            ))

    return ChatResponse(answer=answer.strip(), sources=sources[:3])


@app.get("/health")
async def health():
    return {
        "status"     : "ok",
        "rag_ready"  : rag_chain is not None,
        "llm_model"  : LLM_MODEL,
        "embed_model": EMBED_MODEL,
    }


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
