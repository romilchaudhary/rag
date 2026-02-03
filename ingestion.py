from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

llm = ChatOllama(model="phi3:latest", temperature=0.3)

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

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(retriever, query, prompt_template):
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)
    return response.content

def retrieval_chain_with_lcel(retriever, query, prompt_template):
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context = itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("Ingestion script started...")

    prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:"""
    )

    try:
        vector_store = data_ingestion()
        print("Ingestion script completed successfully.")
    except Exception as e:
        print(f"An error occurred during ingestion: {e}")
        raise e

    print("Start Retrieval Script...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print(retriever)
    query = "what is Pinecone in machine learning?"
    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    """
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(retriever, query, prompt_template)
    print("\nAnswer:")
    print(result_without_lcel)
    """

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = retrieval_chain_with_lcel(retriever, query, prompt_template)
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)

