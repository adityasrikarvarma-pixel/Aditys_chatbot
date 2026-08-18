import logging
import os
import warnings

# Suppress Hugging Face, Transformers & PyTorch Warning Logs
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import ollama
import pdfplumber
import streamlit as st
import streamlit.components.v1 as components

# OCR & Image Processing Dependencies
from pdf2image import convert_from_path
import pytesseract

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Document Intelligence Center", page_icon="⚡", layout="wide"
)

# Custom Styling to match dark 3D aesthetic
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.4);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 2. Interactive 3D Visual Header (Three.js WebGL Engine)
# ------------------------------------------------------------------------------
three_js_header = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { margin: 0; overflow: hidden; background: #0b0f19; }
        #canvas-container { width: 100vw; height: 220px; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="canvas-container"></div>
    <script>
        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.z = 2.8;

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        // Create 3D Particle Cloud
        const particleCount = 1800;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);

        for(let i = 0; i < particleCount * 3; i += 3) {
            const u = Math.random();
            const v = Math.random();
            const theta = u * 2.0 * Math.PI;
            const phi = Math.acos(2.0 * v - 1.0);
            const r = 1.2 + (Math.random() - 0.5) * 0.3;

            positions[i] = r * Math.sin(phi) * Math.cos(theta);
            positions[i+1] = r * Math.sin(phi) * Math.sin(theta);
            positions[i+2] = r * Math.cos(phi);

            // Color gradient (Indigo to Cyan)
            colors[i] = 0.39 + Math.random() * 0.2;
            colors[i+1] = 0.4 + Math.random() * 0.5;
            colors[i+2] = 0.95;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.02,
            vertexColors: true,
            transparent: true,
            opacity: 0.85
        });

        const particles = new THREE.Points(geometry, material);
        scene.add(particles);

        // Interactive Mouse Effect
        let mouseX = 0, mouseY = 0;
        document.addEventListener('mousemove', (e) => {
            mouseX = (e.clientX / window.innerWidth - 0.5) * 0.5;
            mouseY = (e.clientY / window.innerHeight - 0.5) * 0.5;
        });

        // Animation Loop
        function animate() {
            requestAnimationFrame(animate);
            particles.rotation.y += 0.003;
            particles.rotation.x += (mouseY - particles.rotation.x) * 0.05;
            particles.rotation.y += (mouseX - particles.rotation.y) * 0.05;
            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });
    </script>
</body>
</html>
"""

# Render 3D Canvas
components.html(three_js_header, height=220)

st.title("⚡ AI Document Intelligence Hub")
st.caption(
    "Query complex PDF documents locally using vector embeddings and Ollama LLM execution."
)

# ------------------------------------------------------------------------------
# 3. Sidebar Configuration & Controls
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Workspace Controls")
    uploaded_file = st.file_uploader(
        "Upload PDF Document", type=["pdf"], key="pdf_uploader"
    )

    st.divider()

    if st.button("🗑️ Clear Chat Context", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Status:** Local Vectorstore Active")

# ------------------------------------------------------------------------------
# 4. Document Processing Logic
# ------------------------------------------------------------------------------
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

poppler_path = (
    r"C:\poppler\Library\bin"
    if os.path.exists(r"C:\poppler\Library\bin")
    else None
)


@st.cache_resource(show_spinner="Indexing document into Chroma vector DB...")
def process_pdf(file_bytes):
    temp_filename = "temp_uploaded.pdf"
    try:
        with open(temp_filename, "wb") as f:
            f.write(file_bytes)

        docs = []

        try:
            loader = PyPDFLoader(temp_filename)
            docs = loader.load()
        except Exception:
            docs = []

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
                raise RuntimeError(f"OCR Error: {str(ocr_err)}") from ocr_err

        if not docs or not any(doc.page_content.strip() for doc in docs):
            raise ValueError("No readable text could be extracted.")

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
# 5. Interactive Chat & Context Inspection
# ------------------------------------------------------------------------------
if uploaded_file:
    try:
        vectorstore = process_pdf(uploaded_file.getvalue())
        st.success("✅ File indexed. You can now query your document.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_query = st.chat_input("Ask a question about your document...")

        if user_query:
            st.session_state.messages.append(
                {"role": "user", "content": user_query}
            )
            with st.chat_message("user"):
                st.write(user_query)

            # Similarity search
            results = vectorstore.similarity_search(user_query, k=3)
            context = "\n\n".join([doc.page_content for doc in results])

            with st.chat_message("assistant"):
                with st.spinner("Analyzing context..."):
                    try:
                        response = ollama.chat(
                            model="llama3.2",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "Answer strictly based on the provided document context.",
                                },
                                {
                                    "role": "user",
                                    "content": f"Context:\n{context}\n\nQuestion: {user_query}",
                                },
                            ],
                        )

                        answer = response["message"]["content"]
                        st.write(answer)

                        # Source Context Inspection
                        with st.expander(
                            "🔍 View Retrieved Document Chunks (ChromaDB)"
                        ):
                            for idx, doc in enumerate(results, 1):
                                page_num = doc.metadata.get("page", "N/A")
                                st.markdown(
                                    f"**Chunk {idx} (Page {page_num}):**"
                                )
                                st.caption(doc.page_content[:350] + "...")

                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )
                    except Exception as err:
                        st.error(f"Ollama execution error: {str(err)}")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("👈 Upload a PDF document in the sidebar to begin querying.")