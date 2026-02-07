import streamlit as st
import os
import sqlite3
import json
import google.generativeai as genai
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

DB_NAME = "History.db"

def init_db():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            query TEXT,
            response TEXT,
            sources TEXT
        )
    ''')
    conn.commit()
    return conn

def save_log(conn, query, response, sources):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO logs (timestamp, query, response, sources) VALUES (?, ?, ?, ?)",
        (datetime.now(), query, response, json.dumps(sources))
    )
    conn.commit()

st.set_page_config(page_title="Comprehend", layout="wide")
st.title("📄 Comprehend: RAG Based Chat App")
db_conn = init_db()

with st.sidebar:
    st.title("Settings & Logs")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)

    st.divider()
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.checkbox("Show Database Logs"):
        st.subheader("Recent Queries")
        history = db_conn.cursor().execute("SELECT timestamp, query FROM logs ORDER BY id DESC LIMIT 5").fetchall()
        for ts, q in history:
            st.caption(f"{ts}")
            st.text(q[:50] + "...")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

def get_embedding_model():
    try:
        models = [m.name for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
        return "models/gemini-embedding-001" if "models/gemini-embedding-001" in models else models[0]
    except:
        return "models/gemini-embedding-001"

def get_chat_model():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in ["models/gemini-3-flash-preview", "models/gemini-2.5-flash"]:
            if m in models: return m
        return models[0]
    except:
        return "models/gemini-3-flash-preview"

uploaded_file = st.file_uploader("Upload Text Document", type="txt")

if uploaded_file and api_key and st.session_state.vector_store is None:
    with st.spinner("Indexing document..."):
        raw_text = uploaded_file.read().decode("utf-8")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.create_documents([raw_text])
        
        embed_model = get_embedding_model()
        embeddings = GoogleGenerativeAIEmbeddings(model=embed_model)
        st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
        st.success(f"Indexed with {embed_model}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.caption(f"Source {i+1}:")
                    st.write(source)


if prompt := st.chat_input("Ask a question..."):
    if not api_key or not st.session_state.vector_store:
        st.warning("Please provide an API key and upload a file.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            model_name = get_chat_model()
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "Use the context to answer: {context}"),
                ("human", "{input}"),
            ])
            
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

            response = rag_chain.invoke({"input": prompt})
            
            answer = response["answer"]
            source_contents = [doc.page_content for doc in response["context"]]
            
            save_log(db_conn, prompt, answer, source_contents)
            
            st.markdown(answer)
            with st.expander("View Sources"):
                for i, text in enumerate(source_contents):
                    st.caption(f"Source {i+1}:")
                    st.write(text)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "sources": source_contents
            })

