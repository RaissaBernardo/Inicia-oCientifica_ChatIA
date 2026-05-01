"""
ingest.py - Processa o PDF da Carda TC 15 e cria o banco vetorial ChromaDB.

Uso padrão (valores padrão):
    python ingest.py

Uso com parâmetros personalizados (para testes de avaliação):
    python ingest.py --chunk_size 300 --chunk_overlap 30
    python ingest.py --chunk_size 900 --chunk_overlap 90
"""

import sys
import shutil
import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Configuracoes
CHROMA_DIR  = Path(__file__).parent / "chroma_db"
EMBED_MODEL = "nomic-embed-text"

# Valores padrão (usados quando não há argumentos)
CHUNK_SIZE_PADRAO    = 600
CHUNK_OVERLAP_PADRAO = 80


def main():
    # Lê os argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Gera o banco vetorial ChromaDB a partir do PDF.")
    parser.add_argument("--chunk_size",    type=int, default=CHUNK_SIZE_PADRAO,
                        help=f"Tamanho de cada chunk (padrão: {CHUNK_SIZE_PADRAO})")
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP_PADRAO,
                        help=f"Sobreposição entre chunks (padrão: {CHUNK_OVERLAP_PADRAO})")
    args = parser.parse_args()

    CHUNK_SIZE    = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap

    print(f"\n=== INGEST  |  CHUNK_SIZE={CHUNK_SIZE}  CHUNK_OVERLAP={CHUNK_OVERLAP} ===\n")

    # Tenta os dois nomes de arquivo (com e sem cedilha/acento)
    candidates = [
        Path(__file__).parent.parent / "data" / "Carda TC 15_2017_Informação.pdf",
        Path(__file__).parent.parent / "data" / "Carda TC 15_2017_Informacao.pdf",
    ]
    pdf_path = next((p for p in candidates if p.exists()), None)

    if pdf_path is None:
        print("[ERRO] PDF não encontrado. Coloque o arquivo na pasta data/")
        sys.exit(1)

    print(f"[1/4] Carregando PDF: {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    pages  = loader.load()
    print(f"      -> {len(pages)} páginas carregadas.")

    print("[2/4] Dividindo em chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    print(f"      -> {len(chunks)} chunks gerados.")

    print(f"[3/4] Gerando embeddings com '{EMBED_MODEL}'...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    print("[4/4] Salvando no ChromaDB...")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    print(f"\n✅ Banco vetorial salvo em: {CHROMA_DIR}")
    print("   Agora rode:  python -m uvicorn main:app --reload")


if __name__ == "__main__":
    main()
