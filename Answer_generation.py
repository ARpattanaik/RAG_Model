import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


# Load environment variables
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")


persistent_directory = "db/chroma_db"


# --------------------------------------------------
# Load embeddings and vector store
# --------------------------------------------------

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


# --------------------------------------------------
# Search for relevant documents
# --------------------------------------------------

query = "In which year did Tesla begin production of the Roadster?"

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

# Alternative:
#
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3
#     }
# )


relevant_docs = retriever.invoke(query)


print(f"User Query: {query}")

print("\n--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:")
    print(doc.page_content)
    print()


# --------------------------------------------------
# Combine query and retrieved documents
# --------------------------------------------------

combined_input = f"""
Based on the following documents, please answer this question:

Question:
{query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear and helpful answer using only the information
from these documents.

If you cannot find the answer in the documents, say:
"I don't have enough information to answer that question based on
the provided documents."
"""


# --------------------------------------------------
# Create Gemini chat model
# --------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=google_api_key,
    temperature=0
)


# --------------------------------------------------
# Define messages
# --------------------------------------------------

messages = [
    SystemMessage(
        content="You are a helpful assistant. Answer questions using only the provided documents."
    ),
    HumanMessage(content=combined_input),
]


# --------------------------------------------------
# Generate answer
# --------------------------------------------------

result = model.invoke(messages)


print("\n--- Generated Response ---")
print("Content only:")
print(result.content)