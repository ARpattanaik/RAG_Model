import os
import warnings

from dotenv import load_dotenv
from google import genai

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found. Check your .env file."
    )

# Create Gemini client AFTER loading the API key
client = genai.Client(api_key=google_api_key)


# --------------------------------------------------
# 1. Load Documents
# --------------------------------------------------

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory."""

    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist. "
            f"Please create it and add your company files."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .txt files found in {docs_path}. "
            f"Please add your company documents."
        )

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")

    return documents


# --------------------------------------------------
# 2. Split Documents
# --------------------------------------------------

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into smaller chunks."""

    print("\nSplitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    # Development/testing limit

    chunks = chunks[:80]

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


# --------------------------------------------------
# 3. Create Vector Store
# --------------------------------------------------

def create_vector_store(
    chunks,
    persist_directory="db/chroma_db"
):
    """Create and persist ChromaDB vector store."""

    print("\nCreating embeddings and storing in ChromaDB...")

    # Google Gemini embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    print("--- Creating vector store ---")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={
            "hnsw:space": "cosine"
        }
    )

    print("--- Finished creating vector store ---")

    print(
        f"Vector store created and saved to "
        f"{persist_directory}"
    )

    return vectorstore


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Starting RAG document ingestion...\n")

    # 1. Load documents
    documents = load_documents(
        docs_path="docs"
    )

    # 2. Split documents into chunks
    chunks = split_documents(
        documents
    )

    # 3. Create embeddings and store them in ChromaDB
    vectorstore = create_vector_store(
        chunks
    )

    print("\nRAG document ingestion completed successfully.")


if __name__ == "__main__":
    main()