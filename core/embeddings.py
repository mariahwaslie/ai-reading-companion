from sentence_transformers import SentenceTransformer
from .parser import parse_epub
from .chunker import chunk_text

# load model ONCE (important for performance)
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Converts text chunks into embeddings (vectors).
    
    Input:
        chunks: list of text strings

    Output:
        list of embedding vectors
    """

    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings



# -----------------------

# TEST PIPELINE (TEMP)

# -----------------------

# if __name__ == "__main__":

#     # 1. PARSE

#     text = parse_epub("sample_books/pg3296-images-3.epub")

#     # 2. CHUNK

#     chunks = chunk_text(text)

#     print("Number of chunks:", len(chunks))

#     print("Sample chunk:\n", chunks[0][:300])

#     # 3. EMBEDDINGS

#     embeddings = embed_chunks(chunks)

#     print("\nEmbedding shape:")

#     print(len(embeddings), len(embeddings[0]))

#     print("\nFirst embedding sample:")

#     print(embeddings[0][:10])