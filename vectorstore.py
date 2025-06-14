from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()  # Load API key from .env

def create_vector_store():
    loader = TextLoader("docs/maitrii_info.txt")
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()  # Uses OPENAI_API_KEY from .env
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("faiss_index")

if __name__ == "__main__":
    create_vector_store()
