def retrieve_documents(query, faiss_index, k=8):
    """
    Retrieve top-k document chunks most similar to the query.
    """
    docs = faiss_index.similarity_search(query, k=k)
    return [(doc.page_content, doc.metadata) for doc in docs]

def retrieve_documents_per_document(query, faiss_index, k=3):
    """
    Retrieve top-k chunks per document, then combine.
    """
    docs = faiss_index.similarity_search(query, k=30)
    from collections import defaultdict
    source_to_docs = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown document')
        source_to_docs[source].append(doc)
    selected = []
    for doclist in source_to_docs.values():
        selected.extend(doclist[:k])
    return [(doc.page_content, doc.metadata) for doc in selected]