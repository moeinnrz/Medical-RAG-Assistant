# Medical RAG Assistant

A document-grounded Retrieval-Augmented Generation (RAG) application that retrieves relevant passages from a PDF and uses a Groq-hosted LLM to generate answers from that context.

> **Educational project:** This application is for document retrieval and learning purposes. It is not a medical diagnostic system and should not replace professional medical advice.

## Features

- PDF text extraction
- Character-based document chunking
- Semantic embeddings with Sentence Transformers
- Persistent ChromaDB vector storage
- Similarity-based retrieval
- Groq LLM generation
- Source page reporting
- Configurable retrieval count
- Environment-based API configuration
- Colorized CLI

## Architecture

```text
PDF
 |
 v
Text Extraction
 |
 v
Chunking
 |
 v
Sentence Transformer
 |
 v
ChromaDB
 |
 +---- Query Embedding
 |          |
 |          v
 |     Similarity Search
 |          |
 +----------+
            |
            v
     Retrieved Context
            |
            v
        Groq LLM
            |
            v
      Grounded Answer
            |
            v
        Page Sources
```

## Requirements

- Python 3.10+
- A Groq API key
- A readable PDF document

## Installation

```bash
git clone https://github.com/moeinnrz/Medical-RAG-Assistant.git
cd Medical-RAG-Assistant

python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Create `.env` from `.env.example`:

```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
PDF_PATH=symptoms.pdf
CHROMA_DB_PATH=./chroma_db
TOP_K=4
```

Place your source PDF in the project directory and run:

```bash
python main.py
```

## RAG Pipeline

1. Extract text from the PDF.
2. Split text into overlapping chunks.
3. Convert chunks into vector embeddings.
4. Store embeddings in ChromaDB.
5. Embed the user's question.
6. Retrieve the most relevant chunks.
7. Send only retrieved context to the LLM.
8. Return the answer with source page numbers.

## Data and Privacy

The example source PDF is intentionally not included by default. Add only documents that you have permission to use and avoid committing private or sensitive documents.

## Security

Never commit `.env` or real API keys.

## License

This project is provided for educational and portfolio purposes.
