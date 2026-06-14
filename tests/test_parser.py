import unittest
from unittest.mock import patch

from core import parser


class FakePdfPage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakeEpubItem:
    def __init__(self, item_type, content):
        self._item_type = item_type
        self._content = content

    def get_type(self):
        return self._item_type

    def get_content(self):
        return self._content


class FakeEpubBook:
    def __init__(self, items):
        self._items = items

    def get_items(self):
        return self._items


class ParserTests(unittest.TestCase):
    def test_clean_text_collapses_whitespace(self):
        text = "  First line\n\nSecond\tline   with   gaps  "

        self.assertEqual(
            parser.clean_text(text),
            "First line Second line with gaps",
        )

    def test_parse_pdf_normalizes_text_from_each_page(self):
        pages = [
            FakePdfPage("  Chapter  One\n\nstarts here.  "),
            FakePdfPage("Page\t two has    extra spaces."),
        ]
        seen_paths = []

        class FakePdfReader:
            def __init__(self, filepath):
                seen_paths.append(filepath)
                self.pages = pages

        with patch.object(parser, "PdfReader", FakePdfReader):
            result = parser.parse_pdf("book.pdf")

        self.assertEqual(
            result,
            "Chapter One starts here.\n\nPage two has extra spaces.",
        )
        self.assertEqual(seen_paths, ["book.pdf"])

    def test_parse_pdf_skips_pages_without_extractable_text(self):
        pages = [
            FakePdfPage("First page"),
            FakePdfPage(None),
            FakePdfPage("   \n\t   "),
            FakePdfPage("Last page"),
        ]

        class FakePdfReader:
            def __init__(self, filepath):
                self.pages = pages

        with patch.object(parser, "PdfReader", FakePdfReader):
            result = parser.parse_pdf("book.pdf")

        self.assertEqual(result, "First page\n\nLast page")

    def test_parse_epub_extracts_and_cleans_document_items(self):
        items = [
            FakeEpubItem(
                parser.ITEM_DOCUMENT,
                b"<html><body><p>First&nbsp; chapter.</p></body></html>",
            ),
            FakeEpubItem("image", b"<p>This should be ignored.</p>"),
            FakeEpubItem(
                parser.ITEM_DOCUMENT,
                b"<html><body><p>Second\nchapter\ttext.</p></body></html>",
            ),
        ]
        seen_paths = []

        def fake_read_epub(filepath):
            seen_paths.append(filepath)
            return FakeEpubBook(items)

        with patch.object(parser.epub, "read_epub", fake_read_epub):
            result = parser.parse_epub("book.epub")

        self.assertEqual(
            result,
            "First chapter.\n\nSecond chapter text.",
        )
        self.assertEqual(seen_paths, ["book.epub"])

    def test_parse_epub_preserves_spacing_between_html_elements(self):
        items = [
            FakeEpubItem(
                parser.ITEM_DOCUMENT,
                b"<html><body><h1>Chapter One</h1><p>Opening paragraph.</p></body></html>",
            )
        ]

        with patch.object(
            parser.epub,
            "read_epub",
            lambda filepath: FakeEpubBook(items),
        ):
            result = parser.parse_epub("book.epub")

        self.assertEqual(result, "Chapter One Opening paragraph.")

    def test_parse_epub_skips_unreadable_document_items(self):
        items = [
            FakeEpubItem(parser.ITEM_DOCUMENT, b"<p>Readable chapter.</p>"),
            FakeEpubItem(parser.ITEM_DOCUMENT, b"\xff\xfe\xfa"),
            FakeEpubItem(parser.ITEM_DOCUMENT, b"<p>Another readable chapter.</p>"),
        ]

        with patch.object(
            parser.epub,
            "read_epub",
            lambda filepath: FakeEpubBook(items),
        ):
            result = parser.parse_epub("book.epub")

        self.assertEqual(
            result,
            "Readable chapter.\n\nAnother readable chapter.",
        )

    def test_parse_epub_returns_empty_string_when_no_documents(self):
        items = [
            FakeEpubItem("image", b"cover image"),
            FakeEpubItem("style", b"body { font-size: 1rem; }"),
        ]

        with patch.object(
            parser.epub,
            "read_epub",
            lambda filepath: FakeEpubBook(items),
        ):
            result = parser.parse_epub("book.epub")

        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
