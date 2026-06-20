# import chromadb
# from sentence_transformers import SentenceTransformer
# from retriever import retrieve, _extract_filename_and_clean_query

# model = SentenceTransformer("all-MiniLM-L6-v2")
# client = chromadb.PersistentClient(path="./vectordb")
# collection = client.get_collection(name="buitems")

# def debug_retrieve(question, top_k=20):
#     """Debug version showing similarity scores and all results with hybrid search"""
#     print(f"\n🔍 Debugging query: '{question}'")
#     print("=" * 60)
    
#     # Check filename extraction
#     filename_filter, cleaned_question = _extract_filename_and_clean_query(question)
#     print(f"Filename filter: {filename_filter}")
#     print(f"Cleaned question: '{cleaned_question}'")
    
#     # Get embedding
#     question_embedding = model.encode([cleaned_question]).tolist()
    
#     where_clause = {"source": filename_filter} if filename_filter else None
#     if where_clause:
#         print(f"Using where clause: {where_clause}")
    
#     # SEMANTIC SEARCH
#     print("\n[SEMANTIC SEARCH]")
#     results = collection.query(
#         query_embeddings=question_embedding,
#         n_results=top_k,
#         where=where_clause,
#         include=["documents", "distances", "metadatas"]
#     )
    
#     chunks = results.get("documents", [[]])[0]
#     distances = results.get("distances", [[]])[0]
#     metadatas = results.get("metadatas", [[]])[0]
    
#     print(f"Found {len(chunks)} semantic results")
    
#     # KEYWORD SEARCH (Hybrid approach)
#     print("[KEYWORD SEARCH]")
#     # Extract keywords but filter out common question words
#     common_words = {'tell', 'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'me', 'about', 'i', 'you', 'a', 'an', 'the'}
#     keywords = [word for word in cleaned_question.split() 
#                 if word and word[0].isupper() and word.lower() not in common_words]
#     print(f"Extracted keywords (filtered): {keywords}")
    
#     keyword_chunks = []
#     keyword_metadatas = []
#     keyword_distances = []
    
#     if keywords:
#         all_results = collection.get(
#             where=where_clause,
#             include=["documents", "metadatas"]
#         )
#         all_docs = all_results["documents"]
#         all_metas = all_results["metadatas"]
        
#         # Find docs matching keywords
#         for doc, meta in zip(all_docs, all_metas):
#             if any(keyword.lower() in doc.lower() for keyword in keywords):
#                 if doc not in chunks:  # Don't duplicate semantic results
#                     keyword_chunks.append(doc)
#                     keyword_metadatas.append(meta)
#                     keyword_distances.append(0)  # Keyword match = distance 0
        
#         print(f"Found {len(keyword_chunks)} keyword matches")
    
#     # MERGE RESULTS - Put keyword matches FIRST (higher priority)
#     combined_chunks = keyword_chunks + chunks
#     combined_distances = keyword_distances + distances
#     combined_metadatas = keyword_metadatas + metadatas
    
#     # Keep only top_k
#     combined_chunks = combined_chunks[:top_k]
#     combined_distances = combined_distances[:top_k]
#     combined_metadatas = combined_metadatas[:top_k]
    
#     print(f"\n📊 COMBINED RESULTS ({len(combined_chunks)} total):")
#     print("-" * 60)
    
#     for i, (chunk, distance, metadata) in enumerate(zip(combined_chunks, combined_distances, combined_metadatas)):
#         if distance == 0:
#             match_type = "🔍 KEYWORD"
#         else:
#             match_type = "📈 SEMANTIC"
        
#         if distance == 0:
#             print(f"\n[{i+1}] {match_type}")
#         else:
#             similarity = 1 - distance
#             print(f"\n[{i+1}] {match_type} - Similarity: {similarity:.4f}")
        
#         print(f"    Source: {metadata.get('source', 'N/A')}")
#         print(f"    Preview: {chunk[:200]}...")
    
#     # Check total collection size
#     print("\n" + "=" * 60)
#     print(f"Total items in collection: {collection.count()}")
    
#     # List all unique sources
#     all_results = collection.get(include=["metadatas"])
#     sources = set([m["source"] for m in all_results["metadatas"]])
#     print(f"Unique sources in DB: {sources}")

# Test cases
if __name__ == "__main__":
    # Replace with your actual queries
    test_questions = [
        "tell me about vice chancellor",  # Should find relevant info about the vice chancellor
        # Add more test queries here
    ]
    
#     for q in test_questions:
#         if "[" not in q:  # Only run non-placeholder queries
#             debug_retrieve(q, top_k=20)
