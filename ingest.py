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
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Map folder names to readable topic names
topic_map = {
    "admission_process":    "Admissions",
    "BUITEMS_SUB_CAMPUS":   "Sub Campuses",
    "FABS":                 "Faculty of Applied Biosciences (FABS)",
    "fee_structure":        "Fee Structure",
    "FICT":                 "Faculty of Information & Communication Technology (FICT)",
    "FLSI":                 "Faculty of Life Sciences & Informatics (FLSI)",
    "FMS":                  "Faculty of Management Sciences (FMS)",
    "FOE":                  "Faculty of Engineering (FOE)",
    "FSSH":                 "Faculty of Social Sciences & Humanities (FSSH)",
    "general":              "General University Information",
    "scholarship":          "Scholarships"
}

pdf_root = "./pdfs"
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

total_chunks = 0

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