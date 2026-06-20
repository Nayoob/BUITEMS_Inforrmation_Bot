import os
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./vectordb")
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

def chunk_text(text, chunk_size=800, overlap=150):
    """
    Smart chunking that preserves ALL information with semantic boundaries and overlap.
    - First tries to split by paragraphs (preserves context)
    - If a section is too large, splits by sentences
    - Implements overlap to maintain context between chunks
    """
    chunks = []
    
    # Split by double newlines first (section/paragraph boundaries)
    sections = text.split('\n\n')
    
    all_paragraphs = []
    for section in sections:
        # If section is larger than chunk_size, split by sentences
        if len(section) > chunk_size:
            sentences = section.split('. ')
            for sent in sentences:
                if sent.strip():
                    all_paragraphs.append(sent.strip() + '.')
        else:
            if section.strip():
                all_paragraphs.append(section.strip())
    
    # Now create chunks with overlap
    current_chunk = ""
    for para in all_paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        else:
            # Save current chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap (include end of previous chunk)
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:].rsplit('\n', 1)[-1]
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Remove very small chunks and empty ones
    chunks = [c for c in chunks if len(c.strip()) > 50]
    
    return chunks

# Recursively find all PDFs in subdirectories
pdf_folder = "./expdf"
all_pdfs = []

# Collect all PDFs with their folder info
for folder in os.listdir(pdf_root):
    folder_path = os.path.join(pdf_root, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                all_pdfs.append({
                    "path":   os.path.join(folder_path, file),
                    "folder": folder,
                    "file":   file
                })

print(f"Found {len(all_pdfs)} PDFs\n")

if not all_pdfs:
    print("No PDFs found in pdfs/ folder!")
else:
    for idx, pdf_path in enumerate(all_pdfs, 1):
        try:
            # Extract content type from PDF name
            relative_path = os.path.relpath(pdf_path, pdf_folder)
            pdf_name = os.path.basename(pdf_path)
            content_type = pdf_name
            
            print(f"[{idx}/{len(all_pdfs)}] Processing: {relative_path}")

for idx, pdf_info in enumerate(all_pdfs, 1):
    pdf_path = pdf_info["path"]
    folder   = pdf_info["folder"]
    filename = pdf_info["file"]
    topic    = topic_map.get(folder, folder)

    print(f"[{idx}/{len(all_pdfs)}] {folder}/{filename}")
    print(f"  Topic: {topic}")

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

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[f"{folder}_{filename}_chunk_{i}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "topic":    topic,
                "folder":   folder,
                "filename": filename
            }]
        )

    total_chunks += len(chunks)
    print(f"  → Stored ✅\n")

print("=" * 50)
print(f"All PDFs ingested successfully!")
print(f"Total chunks stored: {total_chunks}")
print(f"Total PDFs processed: {len(all_pdfs)}")