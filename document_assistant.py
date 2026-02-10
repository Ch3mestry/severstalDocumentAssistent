import os
from typing import List

from PyPDF2 import PdfReader
from docx import Document


class DocumentAssistant:

    def __init__(self, chunk_size=500, chunk_overlap=50, top_k=3):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.chunks = []

    def _extract_text_from_pdf(self, path: str) -> str:
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)

    def _extract_text_from_docx(self, path: str) -> str:
        doc = Document(path)
        paragraphs = []
        for p in doc.paragraphs:
            if p.text.strip():
                paragraphs.append(p.text)
        return "\n".join(paragraphs)

    def _extract_text(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return self._extract_text_from_pdf(path)
        elif ext in (".docx", ".doc"):
            return self._extract_text_from_docx(path)
        elif ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _split_into_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def index_documents(self, documents: List[str]):
        self.chunks = []
        for doc_path in documents:
            text = self._extract_text(doc_path)
            doc_chunks = self._split_into_chunks(text)
            self.chunks.extend(doc_chunks)
