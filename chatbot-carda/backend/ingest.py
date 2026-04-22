"""
ingest.py - Processa o PDF da Carda TC 15 e cria o banco vetorial ChromaDB.
Execute UMA VEZ antes de rodar o servidor:
    python ingest.py
"""

import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# ── Configurações ──────────────────────────────────────────────
PDF_PATH    = Path(__file__).parent.parent / "data" / "Carda TC 15_2017_Informação.pdf"
CHROMA_DIR  = Path(__file__).parent / "chroma_db"
EMBED_MODEL = "nomic-embed-text"   # modelo leve de embeddings do Ollama
CHUNK_SIZE  = 600
CHUNK_OVERLAP = 80
# ──────────────────────────────────────────────────────────────

def main():
    if not PDF_PATH.exists():
        print(f"[ERRO] PDF não encontrado em: {PDF_PATH}")
        print("Coloque o arquivo na pasta  data/  e rode novamente.")
        sys.exit(1)

    print(f"[1/4] Carregando PDF: {PDF_PATH.name}")
    loader = PyPDFLoader(str(PDF_PATH))
    pages  = loader.load()
    print(f"      → {len(pages)} páginas carregadas.")

    print("[2/4] Dividindo em chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    print(f"      → {len(chunks)} chunks gerados.")

    print(f"[3/4] Gerando embeddings com '{EMBED_MODEL}' (pode demorar na 1ª vez)...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    print("[4/4] Salvando no ChromaDB...")
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)          # recria se já existir

    db = Chroma.from_documents(
        documents        = chunks,
        embedding        = embeddings,
        persist_directory= str(CHROMA_DIR)
    )
    db.persist()
    print(f"\n✅  Banco vetorial salvo em: {CHROMA_DIR}")
    print("    Agora rode:  uvicorn main:app --reload")

if __name__ == "__main__":
    main()
