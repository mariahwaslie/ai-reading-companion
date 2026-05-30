import os
import re 
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from pypdf import PdfReader


def clean_text(text: str):
    # Remove extra whitespace and newlines
    return " ".join(text.split())

def parse_pdf(filepath: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        text = page.extract_text() + "\n"
        if text:
            text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
            pages.append(text) # add to pages 
    return "\n\n".join(pages) # combines all the pages and then returns them 

#parse epub files 
def parse_epub(filepath: str):
    book = epub.read_epub(filepath)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            try:

                html = item.get_content().decode("utf_8")

                soup = BeautifulSoup(html, 'html.parser')
                text = clean_text(soup.get_text())

                chapters.append(text)
                
            except Exception:
                # skip anything broken or non-decodable
                continue
    print(chapters[:100])

    return "\n\n".join(chapters)


if __name__ == "__main__":
    book_text = parse_epub("sample_books/pg3296-images-3.epub")
    # print(f"Got {len(book_text)} characters")
    # print(book_text[:500])  # preview first 100 chars