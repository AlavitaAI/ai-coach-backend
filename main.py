import os
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata  # ✅ Import filter




# ✅ Configure NLTK
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
try:
    nltk.download('punkt', quiet=True)
except:
    print("⚠️ Could not download 'punkt'. Run manually if needed.")

# ✅ Set API Key
os.environ["OPENAI_API_KEY"] = "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"

# ✅ Disable OCR to speed up loading and avoid errors
os.environ["UNSTRUCTURED_OCR_AGENT"] = "none"

doc_folder = "docs"
processed_file_log = "processed_files.txt"
processed_files = set()

# ✅ Load already-processed files
if os.path.exists(processed_file_log):
    with open(processed_file_log, "r") as f:
        processed_files = set(line.strip() for line in f)

docs = []

vectordb = Chroma(
    persist_directory="vector_db",
    embedding_function=OpenAIEmbeddings()
)
count = vectordb._collection.count()
print(f"📊 Vector DB currently holds {count} chunks")

# 📄 Load new PDFs only
for fn in os.listdir(doc_folder):
    if fn.lower().endswith(".pdf") and fn not in processed_files:
        path = os.path.join(doc_folder, fn)
        print(f"📄 Loading: {fn}")
        try:
            loader = UnstructuredPDFLoader(path, mode="elements")
            docs_batch = loader.load()
            docs.extend(docs_batch)
            with open(processed_file_log, "a") as log:
                log.write(fn + "\n")
            print(f"✅ Loaded: {fn} — {len(docs_batch)} elements")
        except Exception as e:
            print(f"❌ Failed to process {fn}: {e}")

print(f"✅ Loaded {len(docs)} new document elements")

if docs:
    # ✂️ Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"🧠 Created {len(chunks)} chunks")

    # Filter out complex metadata
    for chunk in chunks:
        chunk.metadata = filter_complex_metadata(chunk.metadata)
    clean_chunks = chunks       

    # Build or update vector DB
    vectordb = Chroma.from_documents(
        clean_chunks,
        OpenAIEmbeddings(),
        persist_directory="vector_db"
    )
    vectordb.persist()
    print("✅ Vector DB built and persisted!")
else:
    print("⚠️ No new documents to process – vector DB unchanged.")