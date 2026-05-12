from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Chroma DB path
persistent_directory = "db/chroma_db"

# Load existing Chroma database
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "How much did Microsoft pay to acquire GitHub?"
# query = "How do you plant tomatoes in a garden?"

print(f"\nQuery: {query}\n")

print("METHOD 1: Similarity Search (k=3)")

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

docs = retriever.invoke(query)

print(f"Retrieved {len(docs)} documents:\n")

for i, doc in enumerate(docs, 1):
    print(f"Document {i}:")
    print(doc.page_content)
    print()

print("-" * 60)