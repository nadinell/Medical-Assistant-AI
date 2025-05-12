🤖 Medical Assistant IA
Medical Assistant IA is a bilingual Streamlit web application powered by a Large Language Model (LLM) and real-time web search. It helps users get concise medical-related answers in French or English, depending on the input language.

🩺 Features

🌐 Web Search Tool: Uses SerpAPI to provide real-time and relevant information from the web.

🧠 LLM Agent: Based on mistral via Ollama, enhanced with LangChain for better reasoning and tool integration.

🗣️ Language Detection: Automatically detects if your question is in French or English and adapts the response accordingly.

🧾 Conversation History: Maintains a styled history of past interactions with user and assistant chat bubbles.

🎨 Custom UI: Clean, user-friendly interface with custom CSS styling via Streamlit.

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

