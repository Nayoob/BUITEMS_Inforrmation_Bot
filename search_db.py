# import chromadb

# client = chromadb.PersistentClient(path="./vectordb")
# collection = client.get_collection(name="buitems")

# def search_keyword(keyword):
#     """Search for exact keyword in all stored documents"""
#     all_results = collection.get(include=["documents", "metadatas"])
#     documents = all_results["documents"]
#     metadatas = all_results["metadatas"]
    
#     print(f"\n🔎 Searching for '{keyword}' in database...")
#     print("=" * 60)
    
#     matches = []
#     for i, (doc, meta) in enumerate(zip(documents, metadatas)):
#         if keyword.lower() in doc.lower():
#             matches.append((doc, meta))
    
#     if matches:
#         print(f"✅ Found {len(matches)} chunks containing '{keyword}'\n")
#         for idx, (doc, meta) in enumerate(matches, 1):
#             print(f"[{idx}] Source: {meta['source']}")
#             # Find and highlight the keyword
#             idx_in_doc = doc.lower().find(keyword.lower())
#             start = max(0, idx_in_doc - 50)
#             end = min(len(doc), idx_in_doc + len(keyword) + 50)
#             preview = doc[start:end]
#             print(f"    Context: ...{preview}...")
#             print()
#     else:
#         print(f"❌ '{keyword}' NOT found in any documents!")
#         print(f"\nTotal chunks in database: {len(documents)}")
    
#     return matches

# # Search for the person
# if __name__ == "__main__":
#     search_keyword("Poma Panezai")
    
