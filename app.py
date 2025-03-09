from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from flask import Flask, render_template, request, jsonify
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

app = Flask(__name__, static_folder="static", template_folder="templates")

# Set USER_AGENT for WebBaseLoader
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# Load and process data
url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(docs)
texts = [chunk.page_content for chunk in text_chunks]

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "llama-chatbot"
namespace = "web-bot"

# Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize Pinecone Vector Store
vectorstore = PineconeVectorStore(
    index_name=index_name,
    pinecone_api_key=PINECONE_API_KEY,
    embedding=embedding_model,
    namespace=namespace
)

# Check if namespace exists
namespace_exists = False
try:
    existing_namespaces = vectorstore._index.describe_index_stats().get("namespaces", {})
    if namespace in existing_namespaces:
        namespace_exists = True
except Exception as e:
    print(f"Error checking namespace: {e}")

if not namespace_exists:
    vectorstore.add_texts(texts=texts)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = """ 
If you don't know the answer, interact according to your intelligence.

Context: {context}
Question: {question}

Return the helpful answer below and nothing else.
Helpful answer: 
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.8,
    max_tokens=512,
    google_api_key=GOOGLE_API_KEY
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
)

general_chat_responses = {
    "hi": "Hi! How can I assist you today?",
    "hello": "Hello! How can I help?",
    "how are you?": "I'm fine! How can I assist you today?",
    "what's up?": "Not much, just here to help! What do you need?",
    "who are you?": "I'm a helpful AI assistant! Ask me anything.",
}

def format_response(response_text):
    # Ensure the response is structured nicely
    response_text = response_text.replace("* ", "\n- ")  # Convert bullet points
    response_text = response_text.replace("Helpful answer:", "").strip()
    return response_text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()  # Fix: Get JSON data properly
    msg = data.get("query", "").strip().lower()
    
    if not msg:
        return jsonify({"response": "Please enter a question."})

    print("User:", msg)

    if msg in general_chat_responses:
        result = general_chat_responses[msg]
    else:
        try:
            result = qa_chain.run(msg)
            if "I don't have access" in result or "I can't provide" in result:
                result = "I may not have specific details, but I'm happy to help! What do you need?"
            result = format_response(result)
        except Exception as e:
            print(f"Error processing request: {e}")
            result = "Sorry, an error occurred while processing your request."

    print("Response:", result)
    return jsonify({"response": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
