import os
import warnings
 
# Set/filter BEFORE the langchain and huggingface imports
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
os.environ["USER_AGENT"] = USER_AGENT
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
 
from pathlib import Path
from uuid import uuid4
 
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
load_dotenv()
 
chunk_size = 1000
embedding_model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
vectorstore_path = Path(__file__).resolve().parent / "resources" / "vectorstore"
collection_name = "real_estate_docs"
 
llm = None
vector_store = None
 
 
def initialize_components():
    global llm, vector_store
 
    if llm is None:
        print("Initializing LLM...")
        llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=500)
 
    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=embedding_model_name, model_kwargs={"device": "cpu"}
        )
        vector_store = Chroma(
            persist_directory=str(vectorstore_path),
            embedding_function=ef,
            collection_name=collection_name,
        )
 
 
def process_url(urls):
    """Generator: yields status messages while loading, splitting and storing."""
    if isinstance(urls, str):
        urls = [urls]
 
    yield "Initializing components..."
    initialize_components()
 
    yield "Resetting vector store..."
    vector_store.reset_collection()  # avoid duplicates
 
    yield "Loading documents from URLs..."
    try:
        loader = WebBaseLoader(
            web_paths=urls,
            header_template={"User-Agent": USER_AGENT},
        )
        docs = loader.load()
    except Exception as e:
        print(f"Error fetching {urls}: {e!r}")
        raise
 
    # Drop blocked or empty pages so they never reach the vector store
    docs = [
        d for d in docs
        if d.page_content.strip() and "Access Denied" not in d.page_content[:200]
    ]
    if not docs:
        raise ValueError(
            f"No usable content loaded from {urls}. The sites may be blocking "
            "scrapers; try different URLs."
        )
 
    yield f"Splitting {len(docs)} document(s) into chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=chunk_size,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(docs)
 
    yield f"Adding {len(chunks)} chunks to the vector database..."
    uuid = str(uuid4())
    vector_store.add_documents(chunks, ids=[f"{uuid}_{i}" for i in range(len(chunks))])
 
    yield "Done."
 
 
def generate_answer(query):
    initialize_components()
 
    if vector_store._collection.count() == 0:
        raise RuntimeError("You must process URLs first.")
 
    print("Generating answer...")
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever()
    )
    result = chain.invoke({"question": query}, return_only_outputs=True)
    return result.get("answer", ""), result.get("sources", "")
 
 
if __name__ == "__main__":
    urls = [
        "https://en.wikipedia.com/wiki/REIT",
        "https://tradingeconomics.com/united-states/30-year-mortgage-rate/news/588887",
    ]
    for status in process_url(urls):
        print(status)
 
    answer, sources = generate_answer(
        "Who signed the law creating REITs in the US, and when?"
    )
    print(f"answer: {answer}")
    print(f"sources: {sources}")      