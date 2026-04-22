"""
main.py - Servidor FastAPI com RAG para o chatbot da Carda TC 15.
Rodar:  uvicorn main:app --reload  (porta padrão: 8000)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Configurações ──────────────────────────────────────────────
CHROMA_DIR   = Path(__file__).parent / "chroma_db"
EMBED_MODEL  = "nomic-embed-text"
LLM_MODEL    = "llama3"              # troque por "mistral" se preferir
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
TOP_K        = 5                     # nº de chunks recuperados por consulta
# ──────────────────────────────────────────────────────────────

# ── Prompt personalizado ───────────────────────────────────────
PROMPT_TEMPLATE = """Você é um assistente técnico especializado na máquina Carda TC 15 (2017).
Responda em português brasileiro, de forma clara e objetiva.
Use APENAS as informações fornecidas no contexto abaixo.
Se a informação não estiver no contexto, diga: "Não encontrei essa informação na documentação da Carda TC 15."

Contexto extraído do manual:
{context}

Pergunta do operador: {question}

Resposta técnica:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE
)
# ──────────────────────────────────────────────────────────────

app = FastAPI(title="Chatbot Carda TC 15", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Carrega RAG na inicialização ───────────────────────────────
qa_chain = None

@app.on_event("startup")
async def startup():
    global qa_chain
    if not CHROMA_DIR.exists():
        print("[AVISO] Banco vetorial não encontrado. Rode 'python ingest.py' primeiro.")
        return

    print("[startup] Carregando ChromaDB...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

    print("[startup] Conectando ao Ollama LLM...")
    llm = Ollama(
        model=LLM_MODEL,
        temperature=0.1,          # respostas mais determinísticas
        num_predict=512
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": TOP_K}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    print("[startup] ✅  RAG pronto!")

# ── Modelos Pydantic ───────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

class SourceInfo(BaseModel):
    page: int
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]

# ── Endpoints ─────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Pergunta vazia.")

    if qa_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Sistema RAG não inicializado. Rode 'python ingest.py' e reinicie o servidor."
        )

    result  = qa_chain({"query": req.question})
    answer  = result["result"].strip()
    sources = []

    for doc in result.get("source_documents", []):
        sources.append(SourceInfo(
            page   = doc.metadata.get("page", 0) + 1,
            snippet= doc.page_content[:180].replace("\n", " ") + "..."
        ))

    # Remove fontes duplicadas (mesma página)
    seen   = set()
    unique = []
    for s in sources:
        if s.page not in seen:
            seen.add(s.page)
            unique.append(s)

    return ChatResponse(answer=answer, sources=unique[:3])


@app.get("/health")
async def health():
    return {
        "status"     : "ok",
        "rag_ready"  : qa_chain is not None,
        "llm_model"  : LLM_MODEL,
        "embed_model": EMBED_MODEL,
    }


# ── Serve o frontend ───────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
