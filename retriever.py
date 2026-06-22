import chromadb
from sentence_transformers import SentenceTransformer
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./vectordb")
collection = client.get_collection(name="buitems")

def _extract_filename_and_clean_query(question):
    """
    Checks if a query contains a filename pattern like 'in file.pdf'.
    If found, it extracts the filename and returns a cleaned query for semantic search.
    Example: "What is the fee in fees.pdf?" -> ("fees.pdf", "What is the fee?")
    """
    # Regex to find patterns like "in/from/in the <filename>.pdf"
    pattern = r'\b(?:in|from|in the|from the)\s+([a-zA-Z0-9_\-]+\.pdf)\b'
    match = re.search(pattern, question, re.IGNORECASE)
    
    if match:
        filename = match.group(1)
        # Remove the matched part from the question to clean it up
        cleaned_question = re.sub(pattern, '', question, flags=re.IGNORECASE).strip()
        return filename, cleaned_question
    
    return None, question

def _normalize_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def _query_keywords(question):
    stopwords = {
        'tell', 'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are',
        'me', 'about', 'i', 'you', 'a', 'an', 'the', 'and', 'of', 'for',
        'on', 'by', 'with', 'to', 'from', 'in', 'please', 'name', 'names',
        'provide', 'give', 'info', 'information', 'details', 'show',
        'list', 'student', 'students', 'university', 'campus', 'department',
        'faculty', 'program', 'admission', 'fee', 'fees'
    }
    normalized = _normalize_text(question)
    tokens = re.findall(r"\b[a-z0-9]+\b", normalized)
    return [token for token in tokens if token not in stopwords]


def _lexical_search(question, docs, top_k=16):
    normalized_question = _normalize_text(question)
    keywords = _query_keywords(question)
    scored = []

    for doc in docs:
        normalized_doc = _normalize_text(doc)
        score = 0

        if normalized_question and normalized_question in normalized_doc:
            score += 120

        score += sum(8 for keyword in set(keywords) if keyword in normalized_doc)

        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    # return [doc for _, doc in scored[:top_k]]
    return [(score, doc) for score, doc in scored[:top_k]]


# Old retrienver

# def retrieve(question, top_k=8):
#     filename_filter, cleaned_question = _extract_filename_and_clean_query(question)
#     question_embedding = model.encode([cleaned_question]).tolist()
#     where_clause = {"source": filename_filter} if filename_filter else None

#     semantic_results = collection.query(
#         query_embeddings=question_embedding,
#         n_results=top_k,
#         where=where_clause,
#         include=["documents", "distances"],
#     )

#     semantic_docs = semantic_results.get("documents", [[]])[0]
#     semantic_distances = semantic_results.get("distances", [[]])[0]

#     semantic_scores = []
#     for doc, dist in zip(semantic_docs, semantic_distances):
#         if dist is None:
#             continue
#         semantic_scores.append((doc, 1.0 / (1.0 + float(dist))))

#     all_docs = collection.get(where=where_clause, include=["documents"]).get("documents", [])
#     lexical_docs = _lexical_search(cleaned_question, all_docs, top_k=top_k * 2)

#     combined_scores = {}
#     for doc, score in semantic_scores:
#         combined_scores[doc] = combined_scores.get(doc, 0.0) + score
#     for doc in lexical_docs:
#         combined_scores[doc] = combined_scores.get(doc, 0.0) + 5.0

#     ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
#     combined_chunks = [doc for doc, _ in ranked[:top_k]]

#     if not combined_chunks:
#         return semantic_docs

#     return combined_chunks

def retrieve(question, top_k=8):
    filename_filter, cleaned_question = _extract_filename_and_clean_query(question)
    question_embedding = model.encode([cleaned_question]).tolist()
    where_clause = {"source": filename_filter} if filename_filter else None

    # --- 1. Get Semantic Results ---
    semantic_results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
        where=where_clause,
        include=["documents", "distances"],
    )

    semantic_docs = semantic_results.get("documents", [[]])[0]
    semantic_distances = semantic_results.get("distances", [[]])[0]

    semantic_scores = []
    for doc, dist in zip(semantic_docs, semantic_distances):
        if dist is None:
            continue
        # Convert distance to a 0-1 similarity score
        semantic_scores.append((doc, 1.0 / (1.0 + float(dist))))

    # --- 2. Get Lexical Results ---
    all_docs = collection.get(where=where_clause, include=["documents"]).get("documents", [])
    
    # NOTE: Ensure _lexical_search was updated to return (score, doc) tuples!
    lexical_results = _lexical_search(cleaned_question, all_docs, top_k=top_k * 2)

    # --- 3. Blend Scores ---
    combined_scores = {}
    
    # Add Semantic Scores (Weighted at 70%)
    for doc, score in semantic_scores:
        combined_scores[doc] = combined_scores.get(doc, 0.0) + (score * 0.7)

    # Add Normalized Lexical Scores (Weighted at 30%)
    if lexical_results:
        max_lexical = lexical_results[0][0] # The highest lexical score is first
        for score, doc in lexical_results:
            # Normalize the lexical score to be between 0 and 1
            normalized_lex_score = score / max_lexical if max_lexical > 0 else 0
            combined_scores[doc] = combined_scores.get(doc, 0.0) + (normalized_lex_score * 0.3)

    # --- 4. Rank and Return ---
    # Sort the dictionary by the blended scores in descending order
    ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Extract just the document text from the top_k results
    combined_chunks = [doc for doc, _ in ranked[:top_k]]

    # Fallback just in case nothing matched
    if not combined_chunks:
        return semantic_docs

    return combined_chunks