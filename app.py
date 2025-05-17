import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
import tempfile
import os
import requests
import time
import base64

# ------------------ Configuration Updates ------------------
MAX_ITERATIONS = 3
FALLBACK_RESPONSE = "This question is not related to medicine. I'm unable to retrieve current information."

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Medical Assistant IA", layout="centered")
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        background-color: #f5f5f5;
        border: 1px solid #d3d3d3;
        padding: 10px;
        font-size: 16px;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 16px;
        margin-top: 10px;
    }
    .chat-bubble-user {
        background-color: #e0f7fa;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-bubble-ai {
        background-color: #f1f8e9;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-bubble-warning {
        background-color: #ffebee;
        color: #b71c1c;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Session Memory ----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "summaries" not in st.session_state:
    st.session_state.summaries = {}

# ---------------- File Upload and RAG Setup ----------------
st.sidebar.markdown("### 📄 Upload PDF Documents")
uploaded_files = st.sidebar.file_uploader("Upload your medical PDFs", type=["pdf"], accept_multiple_files=True)

all_docs = []
if uploaded_files:
    with st.spinner("📄 Processing uploaded PDFs..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            all_docs.extend(docs)

            # Summarize this specific document
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            llm = Ollama(model="mistral")
            chain = load_qa_chain(llm, chain_type="stuff")
            summary = chain.run(input_documents=chunks, question="Please summarize this document.")
            st.session_state.summaries[uploaded_file.name] = summary

        # Combine all documents for retrieval
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_chunks = splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings()
        st.session_state.vectorstore = FAISS.from_documents(all_chunks, embeddings)

    st.sidebar.success("✅ All PDFs uploaded and indexed.")

# ---------------- Sidebar Summaries ----------------
if st.session_state.summaries:
    st.sidebar.markdown("### 📘 Document Summaries")
    for filename, summary in st.session_state.summaries.items():
        with st.sidebar.expander(filename):
            st.write(summary)
            b64 = base64.b64encode(summary.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="{filename}_summary.txt">📥 Download Summary</a>'
            st.markdown(href, unsafe_allow_html=True)

# ---------------- Web Search Tool ----------------
def web_search(query: str) -> str:
    serp_api_key = st.secrets["SERPAPI_KEY"]
    params = {"q": query, "api_key": serp_api_key}
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=10)
        response.raise_for_status()
        results = response.json()
        if 'answer_box' in results and 'snippet' in results['answer_box']:
            return results['answer_box']['snippet']
        if 'organic_results' in results:
            return "\n".join([res.get('snippet', '') for res in results['organic_results'][:2]])
        return "No relevant results found."
    except Exception as e:
        return f"WEB_SEARCH_ERROR: {str(e)}"

# ---------------- RAG Tool ----------------
def rag_search(query: str) -> str:
    if not st.session_state.vectorstore:
        return "DOCUMENT_ERROR: No PDF uploaded"
    try:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)
        llm = Ollama(model="mistral")
        chain = load_qa_chain(llm, chain_type="stuff")
        return chain.run(input_documents=docs, question=query)
    except Exception as e:
        return f"DOCUMENT_ERROR: {str(e)}"

# ---------------- Agent Setup ----------------
@st.cache_resource
def create_agent():
    tools = [
        Tool(name="WebSearch", func=web_search, description="Useful for general medical knowledge. Prefer this for latest guidelines."),
        Tool(name="DocumentRetrieval", func=rag_search, description="Use ONLY if user uploaded a PDF. Search uploaded document for specific information.")
    ]
    llm = Ollama(model="mistral")
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        max_iterations=MAX_ITERATIONS,
        handle_parsing_errors=True,
        early_stopping_method="generate"
    )

# ---------------- LLM-based Classifier ----------------
def is_question_medical_llm(query: str) -> bool:
    classification_prompt = f"""
    You are a strict classifier.
    If the following question is medical, respond only with: YES
    If not, respond only with: NO

    Question: {query}
    """
    llm = Ollama(model="mistral")
    response = llm.invoke(classification_prompt).strip().upper()
    return response == "YES"

# ---------------- Header ----------------
st.markdown("""
    <div style="background-color:#283747; padding:20px; border-radius:10px; text-align:center">
        <h1 style="color:white">🤖 Medical Assistant IA</h1>
        <p style="color:#D6EAF8;">Upload documents + Search the web – Your smart bilingual medical assistant</p>
    </div>
""", unsafe_allow_html=True)

# ---------------- Chat History ----------------
st.markdown("### 🗂️ Conversation History")
for q, r in st.session_state.history:
    st.markdown(f'<div class="chat-bubble-user"><strong>🧑 You:</strong><br>{q}</div>', unsafe_allow_html=True)
    style = "chat-bubble-warning" if r.strip() == FALLBACK_RESPONSE else "chat-bubble-ai"
    st.markdown(f'<div class="{style}"><strong>🤖 Assistant:</strong><br>{r}</div>', unsafe_allow_html=True)

# ---------------- User Input ----------------
query = st.text_input("💬 Ask your medical question:")

# ---------------- Process Input ----------------
if query:
    with st.spinner("Analyzing..."):
        try:
            language = detect(query)
            placeholder = st.empty()
            if not is_question_medical_llm(query):
                response = FALLBACK_RESPONSE
                placeholder.markdown(
                    f'<div class="chat-bubble-warning"><strong>🤖 Assistant:</strong><br>{response}</div>',
                    unsafe_allow_html=True
                )
            else:
                prompt_template = f"""
                {f"Répondez en français" if language == "fr" else "Answer in English"}:

                Role: Strictly medical assistant. Steps:
                1. If question is NOT medical, respond exactly: '{FALLBACK_RESPONSE}'
                2. For medical questions:
                   a. Use DocumentRetrieval ONLY if PDF exists
                   b. Use WebSearch for general info
                   c. If tools fail, respond: '{FALLBACK_RESPONSE}'
                3. Be concise. No technical details. No disclaimers.

                Question: {query}
                """
                agent = create_agent()
                raw_response = agent.run(prompt_template)
                response = ""
                for word in raw_response.split():
                    response += word + " "
                    placeholder.markdown(
                        f'<div class="chat-bubble-ai"><strong>🤖 Assistant:</strong><br>{response}</div>',
                        unsafe_allow_html=True
                    )
                    time.sleep(0.02)
            st.session_state.history.insert(0, (query, response.strip()))
        except Exception as e:
            st.error(f"System error: {str(e)}")
            response = FALLBACK_RESPONSE
            st.markdown(f'<div class="chat-bubble-warning"><strong>🤖 Assistant:</strong><br>{response}</div>', unsafe_allow_html=True)