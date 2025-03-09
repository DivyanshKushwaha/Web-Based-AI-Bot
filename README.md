# Web Based AI Chatbot using Flask & Gemini AI  

An intelligent chatbot built with **Flask**, **Google Gemini AI**, and **Pinecone** for vector storage. It fetches real-time responses from a web-based knowledge base and provides human-like answers.  

---

## Features  

- **Conversational AI** powered by **Google Gemini AI**  
- **Web Scraping** to extract course-related information  
- **Vector Search** using **Pinecone** for efficient retrieval  
- **Flask Web App** with an interactive chat interface  
- **Custom Responses** for general greetings  

---

## Tech Stack  

- **Backend**: Flask, Google Gemini AI, Pinecone    
- **Embeddings**: GoogleGenerativeAIEmbeddings  
- **Web Scraping**: LangChain WebBaseLoader  

---


---

## Setup & Installation  

- **Clone the repository**  
```bash
git clone https://github.com/yourusername/AI-Chatbot.git
cd AI-Chatbot
```
- **Install requriements**
```bash
pip install -r requirements.txt  
```

- **Create the .env file and update with your API keys**
```bash
GOOGLE_API_KEY=your_google_api_key  
PINECONE_API_KEY=your_pinecone_api_key  

```
- **Run the Flask app**
```bash 
python run app.py
```



## How It Works
- Web Scraping: Extracts text from https://brainlox.com/courses/category/technical
- Chunking: Splits text into smaller meaningful parts
- Vector Embeddings: Stores chunks in Pinecone for fast retrieval
- User Query: Converts input into an embedding and retrieves relevant context
- LLM Processing: Google Gemini AI generates an intelligent response

