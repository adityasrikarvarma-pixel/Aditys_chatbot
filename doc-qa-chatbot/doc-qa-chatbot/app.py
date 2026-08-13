import streamlit as st
import os
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="Free Doc Q&A")
st.title("📚 Free Document Q&A (Direct Groq SDK)")

groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

@st.cache_resource(show_spinner="Processing document...")
def process_pdf(file_bytes):
    temp_filename = "temp_uploaded.pdf"
    try:
        with open(temp_filename, "wb") as f:
            f.write(file_bytes)
        loader = PyPDFLoader(temp_filename)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma.from_documents(documents=splits, embedding=embeddings)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if uploaded_file and groq_api_key:
    try:
        vectorstore = process_pdf(uploaded_file.getvalue())
        client = Groq(api_key=groq_api_key)

        user_query = st.chat_input("Ask something about your document...")
        if user_query:
            # 1. Retrieve top 3 relevant text chunks
            results = vectorstore.similarity_search(user_query, k=3)
            context = "\n\n".join([doc.page_content for doc in results])

            # 2. Query Groq Directly
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Answer questions based only on the provided context."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
                    ]
                )
            st.write("### Answer:")
            st.write(response.choices[0].message.content)

    except Exception:
        st.error("ERROR")