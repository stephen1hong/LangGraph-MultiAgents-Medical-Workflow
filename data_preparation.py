from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load your medical documents (you can use a .txt or multiple files)
loader = TextLoader("my_clinical_notes.txt")  # Or your own clinical text file
documents = loader.load()

# 2. Split into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 3. Embed and index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embedding_model)

# 4. Save locally for use in RAG
db.save_local("medical_docs_db")