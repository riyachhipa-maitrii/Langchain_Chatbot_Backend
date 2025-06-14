from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings())
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

def get_agent_response(user_input):
    return qa_chain.run(user_input)


