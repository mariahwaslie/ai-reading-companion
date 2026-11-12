import uuid

import chromadb
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name = "BAAI/bge-base-en-v1.5", collection_name=None):
        self.model = SentenceTransformer(model_name)
        # in-memory DB
        self.client = chromadb.Client()

        # One collection per VectorStore, holding one book at a time (reset per upload).
        # In-memory Chroma clients in the same process share their collections, and app.py
        # creates one VectorStore per Streamlit session, so each instance gets a unique name;
        # otherwise one user's upload would replace every other user's book.
        self.collection_name = collection_name or f"book_{uuid.uuid4().hex}"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
            )
    def build_index(self, chunks):
        existing = self.collection.get()

        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        embeddings = self.model.encode(
            chunks, 
            show_progress_bar=True, 
            convert_to_numpy=True
            )
        

        self.collection.add(
            documents=chunks, #orgianl text 
            embeddings=embeddings, # meaning vectors 
            ids=[str(i) for i in range(len(chunks))],
            metadatas=[
                {"chunk_num": i} for i in range(len(chunks))
            ]
        )

    def search(self, query, k=5):

        query_embedding = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k

        )

        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "ids": results["ids"][0]

        }
        

        