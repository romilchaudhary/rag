from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings

def main():
    print("Hello from rag!")


if __name__ == "__main__":
    print("Ingestion script started...")
    main()
