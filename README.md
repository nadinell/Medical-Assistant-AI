Medical Assistant IA is a bilingual Streamlit web application powered by a Large Language Model (LLM), Retrieval-Augmented Generation (RAG), and real-time web search. It helps users get concise and context-aware medical-related answers in French or English, depending on the input language.

🩺 Features
🌐 Web Search Tool
Uses SerpAPI to provide real-time and relevant medical information from the web.

📄 PDF Upload with RAG
Upload one or multiple medical PDF documents. The assistant can answer questions based on these documents using retrieval-augmented generation.

✨ Summarization
Summarize each uploaded PDF individually. View each summary in a collapsible dropdown and download it with one click.

🧠 LLM Agent
Powered by the mistral model via Ollama, integrated with LangChain for multi-tool reasoning and document processing.

🗣️ Language Detection
Automatically detects French or English questions and answers accordingly.

💬 Conversation History
Styled chat bubbles track your interactions in a visually clean format.

📥 Streaming Response Output
The assistant "types" answers word by word, mimicking a real-time typing experience.

🎨 Custom UI
Modern interface with user-friendly design and CSS-enhanced elements.

🚀 Getting Started
Prerequisites
Make sure you have the following installed:

Python 3.9+

Ollama installed and running (with the mistral model)

SerpAPI key (get one at serpapi.com)

** Install dependencies:
pip install -r requirements.txt

** Set your SerpAPI key:
Create a .streamlit/secrets.toml file:
SERPAPI_KEY = "your-serpapi-key-here"

** Start the app:
python -m streamlit run app.py

🧪 Tech Stack
Streamlit – For the UI

LangChain – For agent-based reasoning and tool use

Ollama – Local LLM (e.g., mistral)

SerpAPI – Web search API

Langdetect – For language detection

🛡️ Disclaimer
This application is meant for informational purposes only and not a substitute for professional medical advice. Always consult a medical professional for health-related decisions.

