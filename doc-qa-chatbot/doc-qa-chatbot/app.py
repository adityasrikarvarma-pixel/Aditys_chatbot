import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Free Document Q&A Chatbot")
st.title("📚 Free Document Q&A Assistant")

# Sidebar for Groq API Key
groq_api_key = st.sidebar.text_input("Groq API Key (Free)", type="password", key="groq_key_input")
st.sidebar.markdown("[Get a free Groq key here](https://console.groq.com/keys)")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Cache vector store setup
@st.cache_resource(show_spinner="Processing document with local embeddings...")
def process_pdf(file_bytes, key_hash):
    temp_filename = "temp_uploaded_doc.pdf"
    try:
        with open(temp_filename, "wb") as f:
            f.write(file_bytes)

        loader = PyPDFLoader(temp_filename)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        return vectorstore
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if uploaded_file and groq_api_key:
    try:
        vectorstore = process_pdf(uploaded_file.getvalue(), hash(uploaded_file.name))
        retriever = vectorstore.as_retriever()

        template = """You are an assistant for question-answering tasks. 
        Use the following pieces of context to answer the question. 
        If you don't know the answer, say you don't know. Keep it concise.

        Context:
        {context}

        Question: {question}
        Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="llama-3.1-8b-instant", 
            temperature=0
        )

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        user_query = st.chat_input("Ask something about your document...")
        if user_query:
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(user_query)
            st.write("### Answer:")
            st.write(response)

    except Exception:
        # Generic error handler to mask all internal exception details
        st.error("ERROR")

elif not groq_api_key:
    st.info("Please enter your free Groq API key in the sidebar to get started.")