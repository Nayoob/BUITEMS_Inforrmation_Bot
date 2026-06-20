import os
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./vectordb")

# To ensure a fresh start, delete the existing collection if it exists
print("Checking for existing collection...")
if "buitems" in [c.name for c in client.list_collections()]:
    print("Deleting existing 'buitems' collection.")
    client.delete_collection(name="buitems")

collection = client.get_or_create_collection(name="buitems")

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                cleaned = "\n".join([
                    line.strip() for line in page_text.splitlines()
                    if line.strip()
                ])
                text += cleaned + "\n"
    return text

def chunk_text(text, chunk_size=1500, overlap=300):
    """
    Smart chunking that respects semantic boundaries (paragraphs, sections).
    Tries to keep related bullet points together.
    """
    chunks = []
    
    # Split by double newlines first (section/paragraph boundaries)
    sections = text.split('\n\n')
    
    current_chunk = ""
    for section in sections:
        # If adding this section keeps us under chunk_size, add it
        if len(current_chunk) + len(section) < chunk_size:
            current_chunk += section + "\n\n"
        else:
            # If current chunk has content, save it
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # Start new chunk with this section
            current_chunk = section + "\n\n"
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Remove empty chunks
    chunks = [c for c in chunks if c.strip()]
    
    return chunks

# Recursively find all PDFs in subdirectories
pdf_folder = "./pdfs"
all_pdfs = []

for root, dirs, files in os.walk(pdf_folder):
    for f in files:
        if f.endswith(".pdf"):
            pdf_path = os.path.join(root, f)
            all_pdfs.append(pdf_path)

print(f"Found {len(all_pdfs)} PDFs\n")

if not all_pdfs:
    print("No PDFs found in pdfs/ folder!")
else:
    for idx, pdf_path in enumerate(all_pdfs, 1):
        try:
            # Extract content type from subdirectory name
            relative_path = os.path.relpath(pdf_path, pdf_folder)
            content_type = os.path.dirname(relative_path)
            pdf_name = os.path.basename(pdf_path)
            
            print(f"[{idx}/{len(all_pdfs)}] Processing: {relative_path}")

            text = extract_text(pdf_path)

            if not text.strip():
                print(f"  ⚠️  No text found, skipping\n")
                continue

            chunks = chunk_text(text)
            print(f"  → {len(chunks)} chunks created")

            embeddings = model.encode(
                chunks,
                batch_size=32,
                show_progress_bar=False
            ).tolist()

            ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source": pdf_name,
                    "content_type": content_type
                } 
                for _ in chunks
            ]

            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )

            print(f"  → Stored ✅\n")
        except Exception as e:
            print(f"  ❌ Error processing {relative_path}: {e}\n")
            continue

print("=" * 50)
print("All PDFs ingested successfully!")
print(f"Total chunks in DB: {collection.count()}")