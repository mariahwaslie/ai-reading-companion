from langchain_text_splitters import RecursiveCharacterTextSplitter
from parser import parse_epub  # or parse_epub

def chunk_text(text: str, chunk_size: int = 800, overlap = 100 ) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len
    )
    texts = text_splitter.split_text(text)
    return texts

if __name__ == "__main__":

    text = parse_epub("sample_books/pg3296-images-3.epub")

    chunks = chunk_text(text)

    print("Total chunks:", len(chunks))

    print("\n--- FIRST CHUNK ---\n")
    print(chunks[0])

    print("\n--- SECOND CHUNK ---\n")
    print(chunks[1])

    print("\n--- LAST CHUNK ---\n")
    print(chunks[-1])
    for i, c in enumerate(chunks[:5]):
        print(f"\nCHUNK {i} LENGTH: {len(c)}")
        print(c[:200])