import os

from constants import GOOGLE_API_KEY
from constants import CHUNK_SIZE, CHUNK_OVERLAP, K
from constants import ARXIV_CODES, TAGS_TO_IGNORE, TAGS_TO_EXTRACT, CHROMA_PERSIST_DIR

# set the google API keys for vision APIs
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def scrape_documents():
    # create a list of urls from the arxiv codes
    urls = [f"https://ar5iv.labs.arxiv.org/html/{arxiv_code}" for arxiv_code in ARXIV_CODES]

    print("Scraping documents from the following URLs:")
    for url in urls:
        print(url)
    
    # create a loader
    loader = AsyncChromiumLoader(urls)

    # create a transformer
    transformer = BeautifulSoupTransformer()

    # scrape the documents
    docs = loader.load()
    docs = transformer.transform_documents(
        docs,
        tags_to_ignore=TAGS_TO_IGNORE,
        tags_to_extract=TAGS_TO_EXTRACT
    )

    return docs

def extract_retriever():
    # scrape the documents
    docs = scrape_documents()

    # create a splitter
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # split the documents
    docs = splitter.split_documents(docs)

    print(f"Extracted {len(docs)} chunks from the documents.")

    # create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_document"
    )

    # check if the persist directory exists
    if os.path.exists(CHROMA_PERSIST_DIR):
        print("Retriever already exists, loading from disk.")
        return Chroma.from_persist_dir(CHROMA_PERSIST_DIR).as_retriever(search_kwargs={"k": K})
    
    else:
        print("Retriever does not exist, creating a new one.")
        return Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PERSIST_DIR).as_retriever(search_kwargs={"k": K})

    