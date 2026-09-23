import chromadb
from sentence_transformers import SentenceTransformer
from knowledge_base import KNOWLEDGE_BASE

# Load the embedding model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up a local, in-memory ChromaDB collection
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="mental_health_kb")

# Only populate the collection once (avoid duplicate inserts on every rerun)
if collection.count() == 0:
    for doc in KNOWLEDGE_BASE:
        embedding = embedding_model.encode(doc["content"]).tolist()
        collection.add(
            ids=[doc["id"]],
            embeddings=[embedding],
            documents=[doc["content"]]
        )


def retrieve_relevant_context(query, top_k=2):
    """Find the most relevant knowledge base entries for a user's question."""
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    retrieved_docs = results["documents"][0]
    return "\n\n".join(retrieved_docs)