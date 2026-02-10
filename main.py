import json
import os
import requests

from document_assistant import DocumentAssistant


DOCUMENT_URLS = {
    "A9RD3D4.pdf": "https://storage.yandexcloud.net/sever-images/severstal/A9RD3D4.pdf",
    "Polzovatelskoe_soglashenie.pdf": "https://storage.yandexcloud.net/sever-images/severstal/Polzovatelskoe_soglashenie.pdf",
    "University Success.docx": "https://storage.yandexcloud.net/sever-images/severstal/University%20Success.docx",
}


def download_documents(base_dir):
    docs_dir = os.path.join(base_dir, "documents")
    os.makedirs(docs_dir, exist_ok=True)

    paths = []
    for filename, url in DOCUMENT_URLS.items():
        filepath = os.path.join(docs_dir, filename)
        if not os.path.exists(filepath):
            print(f"Скачиваю {filename}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(response.content)
        paths.append(filepath)

    return paths


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    documents = download_documents(base_dir)

    assistant = DocumentAssistant(chunk_size=300, chunk_overlap=30, top_k=3)

    print("Индексация документов...")
    assistant.index_documents(documents)
    print(f"Проиндексировано чанков: {len(assistant.chunks)}")

    queries = [
        "Какие обязанности у пользователя согласно пользовательскому соглашению?",
        "Что такое University Success?",
        "Какие условия использования сервиса описаны в документах?",
        "What skills are important for university success?",
        "Какие ограничения ответственности указаны в соглашении?",
    ]

    results = []
    for query in queries:
        print(f"\nВопрос: {query}")
        answer = assistant.answer_query(query)
        print(f"Ответ: {answer}")
        results.append({
            "query": query,
            "answer": answer
        })

    output_path = os.path.join(base_dir, "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nРезультаты сохранены в {output_path}")


if __name__ == "__main__":
    main()
