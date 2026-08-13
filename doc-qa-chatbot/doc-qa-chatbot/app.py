import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Document Q&A Chatbot")
st.title("📚 Document Q&A Assistant")

# Sidebar API Key with unique key
api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key_input")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Cache the vector store setup so it doesn't re-index on every question typed
@st.cache_resource(show_spinner="Processing document...")
def process_pdf(file_bytes, key_hash):
    # Save temporary file safely
    temp_filename = "temp_uploaded_doc.pdf"
    with open(temp_filename, "wb") as f:
        f.write(file_bytes)

    # Load & Split
    loader = PyPDFLoader(temp_filename)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Vector store
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    
    # Cleanup temp file
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    return vectorstore

if uploaded_file and api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    # Get or create cached retriever
    vectorstore = process_pdf(uploaded_file.getvalue(), hash(uploaded_file.name))
    retriever = vectorstore.as_retriever()

    # Define Chain
    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of context to answer the question. 
    If you don't know the answer, say you don't know. Keep it concise.

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

elif not api_key:
    st.info("Please enter your OpenAI API key in the sidebar to get started.")