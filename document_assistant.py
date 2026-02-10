import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document


class DocumentAssistant:

    def __init__(self, chunk_size=500, chunk_overlap=50, top_k=3, model_name="all-MiniLM-L6-v2"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.chunks = []
        self.embeddings = None
        self.model = SentenceTransformer(model_name)

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

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        norm_a = a / np.linalg.norm(a)
        norm_b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(norm_b, norm_a)

    def _find_relevant_chunks(self, query: str) -> List[str]:
        query_embedding = self.model.encode(query)
        scores = self._cosine_similarity(query_embedding, self.embeddings)
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        return [self.chunks[i] for i in top_indices]

    def index_documents(self, documents: List[str]):
        self.chunks = []
        for doc_path in documents:
            text = self._extract_text(doc_path)
            doc_chunks = self._split_into_chunks(text)
            self.chunks.extend(doc_chunks)
        self.embeddings = self.model.encode(self.chunks)

    def _build_prompt(self, query: str, chunks: List[str]) -> str:
        context = "\n\n---\n\n".join(chunks)
        prompt = (
            f"Используй только следующие фрагменты документов для ответа:\n\n"
            f"{context}\n\n"
            f"Вопрос: {query}\n"
            f"Ответ:"
        )
        return prompt

    def _call_llm(self, prompt: str) -> str:
        # Для подключения реальной LLM замените тело этого метода.
        # Например, можно использовать transformers pipeline:
        #   from transformers import pipeline
        #   generator = pipeline("text-generation", model="your-model")
        #   return generator(prompt, max_new_tokens=256)[0]["generated_text"]
        #
        # Или OpenAI API:
        #   import openai
        #   response = openai.ChatCompletion.create(
        #       model="gpt-3.5-turbo",
        #       messages=[{"role": "user", "content": prompt}]
        #   )
        #   return response.choices[0].message.content

        lines = prompt.split("\n")
        answer_parts = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Используй") and not line.startswith("Вопрос:") and not line.startswith("Ответ:") and line != "---":
                answer_parts.append(line)
        if answer_parts:
            return " ".join(answer_parts[:3])
        return "Не удалось найти ответ в предоставленных документах."

    def answer_query(self, query: str) -> str:
        relevant_chunks = self._find_relevant_chunks(query)
        prompt = self._build_prompt(query, relevant_chunks)
        return self._call_llm(prompt)
