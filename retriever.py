import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./vectordb")
collection = client.get_collection(name="buitems")

def retrieve(question, top_k=3):
    question_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    enriched_chunks = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        topic    = meta.get("topic", "General")
        filename = meta.get("filename", "unknown")
        short_doc = doc[:400]
        enriched = f"[Source: {topic} | File: {filename}]\n{short_doc}"
        enriched_chunks.append(enriched)

    return enriched_chunks