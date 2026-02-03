from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

def data_ingestion():
    loader = TextLoader('mediumblog1.txt', encoding='utf-8')
    documents = loader.load()
    print("splitting documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    print(text_splitter)
    docs = text_splitter.split_documents(documents)
    print("Creating embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("Connecting to Pinecone...")
    vector_store = PineconeVectorStore.from_documents(docs, embeddings, index_name="rag-demo")
    print("Ingestion complete.")
    return vector_store


if __name__ == "__main__":
    print("Ingestion script started...")
    try:
        data_ingestion()
        print("Ingestion script completed successfully.")
    except Exception as e:
        print(f"An error occurred during ingestion: {e}")
