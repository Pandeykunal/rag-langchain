from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

load_dotenv()

# Setup
persistent_directory = "db/chroma_db"

# HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# GROQ LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# Load Chroma DB
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Pydantic Model

class QueryVariations(BaseModel):
    queries: List[str]

# MAIN EXECUTION

# Original query
original_query = "How does Tesla make money?"

print(f"\nOriginal Query: {original_query}\n")

# Step 1: Generate Query Variations
llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""
Generate 3 different variations of this query
that would help retrieve relevant documents.

Original Query:
{original_query}

Return only 3 alternative queries.
"""

response = llm_with_tools.invoke(prompt)

query_variations = response.queries

print("Generated Query Variations:\n")

for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "=" * 60)

# Step 2: Retrieve Documents for Each Query
retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

all_retrieval_results = []

for i, query in enumerate(query_variations, 1):

    print(f"\n=== RESULTS FOR QUERY {i} ===")
    print(f"Query: {query}\n")

    docs = retriever.invoke(query)

    # Store results for future RRF
    all_retrieval_results.append(docs)

    print(f"Retrieved {len(docs)} documents:\n")

    for j, doc in enumerate(docs, 1):

        print(f"Document {j}:")
        print(doc.page_content[:150])
        print()

    print("-" * 50)

print("\n" + "=" * 60)
print("Multi-Query Retrieval Complete!")

# Example Structure of Stored Results
"""
all_retrieval_results = [

    [Doc1, Doc2, Doc3, Doc4, Doc5],  ← Query 1 results

    [Doc2, Doc1, Doc6, Doc7, Doc3],  ← Query 2 results

    [Doc8, Doc2, Doc9, Doc10, Doc11] ← Query 3 results
]
"""