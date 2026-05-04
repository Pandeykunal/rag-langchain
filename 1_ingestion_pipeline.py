import os, shutil
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_PATH   = "docs"
PERSIST_DIR = "db/chroma_db"

# Local model — no API key, no rate limits, runs on CPU
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


def load_and_split():
    # Load all .txt files from docs/
    docs = DirectoryLoader(
        DOCS_PATH, glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    ).load()

    if not docs:
        raise FileNotFoundError(f"No .txt files found in '{DOCS_PATH}'.")

    print(f"Loaded {len(docs)} document(s).")

    # Split into chunks — respects paragraph/sentence boundaries
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    ).split_documents(docs)

    print(f"Total chunks: {len(chunks)}")
    return chunks


def main():
    print("=== 🚀 RAG INGESTION PIPELINE ===\n")

    # If DB exists and has data, skip re-ingestion
    if os.path.exists(PERSIST_DIR):
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=EMBEDDING_MODEL)
        count = db._collection.count()

        if count > 0:
            print(f"✅ Existing DB loaded — {count} chunks ready.")
            return db

        # DB is empty/corrupt — wipe and rebuild
        print("⚠️  DB empty. Rebuilding...")
        shutil.rmtree(PERSIST_DIR)

    # Build fresh DB
    chunks = load_and_split()

    try:
        db = Chroma.from_documents(
            chunks, EMBEDDING_MODEL,
            persist_directory=PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)  # Clean up partial DB on failure
        raise RuntimeError(f"❌ Ingestion failed: {e}") from e

    print(f"\n✅ Done! {db._collection.count()} chunks stored at '{PERSIST_DIR}'.")
    return db


if __name__ == "__main__":
    vectorstore = main()