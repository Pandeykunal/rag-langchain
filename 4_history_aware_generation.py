from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os

# Load environment variables
load_dotenv()

# Connect to vector database
persistent_directory = "db/chroma_db"

# Free local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma DB
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

# Free LLM using Groq
model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Store chat history
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Rewrite question using history
    if chat_history:

        messages = [
            SystemMessage(
                content="Rewrite the new question into a standalone searchable question using chat history. Only return the rewritten question."
            ),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()

        print(f"Searching for: {search_question}")

    else:
        search_question = user_question

    # Step 2: Retrieve documents
    retriever = db.as_retriever(search_kwargs={"k": 2})

    docs = retriever.invoke(search_question)

    print(f"\nFound {len(docs)} relevant documents:\n")

    for i, doc in enumerate(docs, 1):

        preview = doc.page_content[:200]

        print(f"Doc {i}:")
        print(preview)
        print()

    # Step 3: Build context
    context = "\n".join([
        doc.page_content[:500]
        for doc in docs
    ])

    combined_input = f"""
Answer the question using ONLY the context below.

Question:
{user_question}

Context:
{context}

If the answer is not found in the context, say:
"I don't have enough information."
"""

    # Step 4: Generate answer
    messages = [
        SystemMessage(
            content="You are a helpful RAG assistant."
        ),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)

    answer = result.content

    # Step 5: Save conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print("\n--- Answer ---")
    print(answer)

    return answer


# Chat loop
def start_chat():

    print("Ask me questions!")
    print("Type 'quit' to exit.\n")

    while True:

        question = input("Your question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()