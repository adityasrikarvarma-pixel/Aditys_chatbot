import os
import pdfplumber
import streamlit as st
import ollama

# OCR & Image Processing Dependencies
from pdf2image import convert_from_path
import pytesseract

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------------------------------------
# 1. Path Configurations
# ------------------------------------------------------------------------------
# Tesseract Path
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Poppler Path
poppler_path = (
    r"C:\poppler\Library\bin"
    if os.path.exists(r"C:\poppler\Library\bin")
    else None
)

# ------------------------------------------------------------------------------
# 2. UI Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Free Doc Q&A (Local Ollama)", page_icon="📚", layout="centered"
)
st.title("📚 Local Document Q&A (Powered by Ollama)")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")


# ------------------------------------------------------------------------------
# 3. Document Processing
# ------------------------------------------------------------------------------
@st.cache_resource(
    show_spinner="Processing document (running OCR if scanned)..."
)
def process_pdf(file_bytes):
    temp_filename = "temp_uploaded.pdf"
    try:
        with open(temp_filename, "wb") as f:
            f.write(file_bytes)

        docs = []

        # Strategy 1: Standard PyPDFLoader
        try:
            loader = PyPDFLoader(temp_filename)
            docs = loader.load()
        except Exception:
            docs = []

        # Strategy 2: pdfplumber Fallback
        if not docs or not any(doc.page_content.strip() for doc in docs):
            docs = []
            try:
                with pdfplumber.open(temp_filename) as pdf:
                    for page_idx, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text and text.strip():
                            docs.append(
                                Document(
                                    page_content=text,
                                    metadata={"page": page_idx + 1},
                                )
                            )
            except Exception:
                docs = []

        # Strategy 3: Tesseract OCR + Poppler Fallback
        if not docs or not any(doc.page_content.strip() for doc in docs):
            docs = []
            try:
                images = convert_from_path(
                    temp_filename, poppler_path=poppler_path
                )
                for i, image in enumerate(images):
                    ocr_text = pytesseract.image_to_string(image)
                    if ocr_text and ocr_text.strip():
                        docs.append(
                            Document(
                                page_content=ocr_text,
                                metadata={"page": i + 1},
                            )
                        )
            except Exception as ocr_err:
                raise RuntimeError(
                    f"OCR Extraction Error: {str(ocr_err)}\n"
                    f"Poppler Path Exists: {os.path.exists(r'C:\\poppler\\Library\\bin')}\n"
                    f"Tesseract Exists: {os.path.exists(tesseract_path)}"
                ) from ocr_err

        if not docs or not any(doc.page_content.strip() for doc in docs):
            raise ValueError(
                "Could not extract any readable text from this PDF file."
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma.from_documents(documents=splits, embedding=embeddings)

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# ------------------------------------------------------------------------------
# 4. Main App Flow
# ------------------------------------------------------------------------------
if uploaded_file:
    try:
        vectorstore = process_pdf(uploaded_file.getvalue())
        st.success("✅ Document processed successfully!")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Render chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_query = st.chat_input("Ask something about your document...")

        if user_query:
            st.session_state.messages.append(
                {"role": "user", "content": user_query}
            )
            with st.chat_message("user"):
                st.write(user_query)

            # Similarity search from Chroma vector store
            results = vectorstore.similarity_search(user_query, k=3)
            context = "\n\n".join([doc.page_content for doc in results])

            with st.chat_message("assistant"):
                with st.spinner("Ollama is generating an answer..."):
                    try:
                        # Call local Ollama instance
                        response = ollama.chat(
                            model="llama3.2",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a helpful assistant. "
                                        "Answer questions strictly based on the provided context."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": f"Context:\n{context}\n\nQuestion: {user_query}",
                                },
                            ],
                        )

                        answer = response["message"]["content"]
                        st.write(answer)

                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )
                    except Exception as ollama_err:
                        st.error(
                            f"Ollama Connection Error: {str(ollama_err)}\n\n"
                            "Make sure Ollama is running on your PC and you pulled the model (`ollama pull llama3.2`)."
                        )

    except Exception as e:
        st.error("An error occurred during processing:")
        st.exception(e)