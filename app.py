from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from prompt import chat_prompt

load_dotenv()

app = Flask(__name__)
CORS(app)  # Temporarily allow all origins for debugging

# Check for OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

print("Loaded API Key:", OPENAI_API_KEY)

# Load FAISS vector store with error handling
try:
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=retriever
    )
except Exception as e:
    print(f"Error loading vectorstore or initializing QA chain: {e}")
    vectorstore = None
    qa = None

@app.route('/chat', methods=['POST'])  # 🔄 Endpoint should match frontend
def chat():
    if qa is None:
        return jsonify({"response": "AI backend is not ready. Please try again later."}), 503
    data = request.get_json()
    user_input = data.get('message', '')

    if not user_input:
        return jsonify({"response": "No message provided."}), 400

    try:
        formatted_prompt = chat_prompt.format(input=user_input)
        response = qa.run(formatted_prompt)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error during QA chain run: {e}")
        return jsonify({"response": "Sorry, there was an error processing your request. Please try again later."}), 500

@app.route('/')
def index():
    return "AI Chatbot Backend is Running!"

if __name__ == '__main__':
    print("WARNING: Running in debug mode. Do not use debug=True in production!")
    app.run(host='0.0.0.0', port=5000)
