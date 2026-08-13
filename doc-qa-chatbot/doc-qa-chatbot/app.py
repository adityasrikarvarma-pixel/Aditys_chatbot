import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Corrected imports for retrieval chains in modern LangChain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="Document Q&A Chatbot")
st.title("📚 Document Q&A Assistant")

# API Key Input
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file and api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load & Split Document
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Store in Vector Database
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # Build QA Chain
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # User Query Input
    user_query = st.chat_input("Ask something about your document...")
    if user_query:
        response = rag_chain.invoke({"input": user_query})
        st.write("### Answer:")
        st.write(response["answer"])
elif not api_key:
    st.info("Please enter your OpenAI API key in the sidebar to get started.")