import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")

persistent_directory = "db/chroma_db"

# Load the same embedding model used during indexing
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# Load existing ChromaDB
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

print("\n--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}:")
    print(doc.page_content)
# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"