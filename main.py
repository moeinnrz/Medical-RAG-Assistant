import os
import re
from pathlib import Path

import chromadb
from colorama import Fore, Style, init
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()
init(autoreset=True)

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
PDF_PATH = os.getenv("PDF_PATH", "symptoms.pdf")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
TOP_K = int(os.getenv("TOP_K", "4"))

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

client = Groq(api_key=API_KEY)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def extract_pages(pdf_path: str):
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\\s+", " ", text).strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text,
            })

    return pages


def chunk_text(text: str, size: int = 800, overlap: int = 100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def build_documents(pdf_path: str):
    documents = []
    metadatas = []
    ids = []

    for page in extract_pages(pdf_path):
        for index, chunk in enumerate(chunk_text(page["text"])):
            documents.append(chunk)
            metadatas.append({"page": page["page"]})
            ids.append(f"page-{page['page']}-chunk-{index}")

    return documents, metadatas, ids


def get_collection():
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return db.get_or_create_collection(name="medical_documents")


def index_pdf(pdf_path: str, collection):
    if collection.count() > 0:
        return

    documents, metadatas, ids = build_documents(pdf_path)

    if not documents:
        raise ValueError("No readable text was found in the PDF.")

    embeddings = embedder.encode(documents).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def retrieve(question: str, collection):
    query_embedding = embedder.encode([question]).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    return list(zip(documents, metadatas))


def answer_question(question: str, retrieved):
    if not retrieved:
        return "I could not find relevant information in the indexed document."

    context_parts = []
    sources = []

    for document, metadata in retrieved:
        page = metadata.get("page", "?")
        context_parts.append(f"[Page {page}] {document}")
        sources.append(str(page))

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document-grounded medical information assistant.

Answer the user's question using only the provided document context.
If the context does not contain enough information, clearly say that the
document does not provide enough information.

Do not diagnose the user and do not invent medical facts.

Document context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Be accurate, concise, and grounded in the supplied context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()
    unique_sources = ", ".join(dict.fromkeys(sources))

    return f"{answer}\n\nSources: PDF page(s) {unique_sources}"


def main():
    print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 58)
    print("                 MEDICAL RAG ASSISTANT")
    print("=" * 58 + Style.RESET_ALL)

    if not Path(PDF_PATH).exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}. Put the source document in the project folder."
        )

    collection = get_collection()
    index_pdf(PDF_PATH, collection)

    print(Fore.WHITE + "Ask questions about the indexed document.")
    print(Fore.WHITE + "Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input(Fore.BLUE + "You > " + Style.RESET_ALL).strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print(Fore.CYAN + "Goodbye!")
            break

        try:
            print(Fore.YELLOW + "Searching document...")
            retrieved = retrieve(question, collection)

            print(Fore.YELLOW + "Generating grounded answer...")
            answer = answer_question(question, retrieved)

            print(Fore.GREEN + f"Assistant > {answer}\n")
        except Exception as exc:
            print(Fore.RED + f"Error: {exc}\n")


if __name__ == "__main__":
    main()
