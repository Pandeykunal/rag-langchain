import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from langchain_core.runnables import RunnablePassthrough     

load_dotenv()

# 1. Embedding Model (local)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load Vector DB
db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embedding_model
)
retriever = db.as_retriever(search_kwargs={"k": 3})

# 3. Load LLM (Groq)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")  
)

# 4. Prompt Template
prompt_template = """
Use the following context to answer the question.
If you don't know the answer, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 5. RAG Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Query
query = "How much did Microsoft pay to acquire GitHub?"
response = qa_chain.invoke(query)

print("\n🧠 Answer:")
print(response)