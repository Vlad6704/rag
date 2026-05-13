# Local RAG with PDF Support

This project is a local Retrieval Augmented Generation (RAG) system designed to digest PDF documentation and allow you to chat with it. It focuses on intelligent chunking of technical documents (like PostgreSQL manuals) to preserve context and code blocks.

It uses **Sentence Transformers** for embeddings and **Ollama** for the local LLM inference.

## Features

- **Smart PDF Chunking**: Converts PDFs to Markdown and splits them by logical sections (chapters, sub-sections).
- **Context Preservation**: Long sections are split into parts while keeping headers and context from previous parts.
- **Code Block Aware**: Prevents splitting code blocks across chunks.
- **Local & Private**: Runs entirely locally using Ollama and local embedding models.

## Prerequisites

1. **Python 3.10+**
2. **Ollama**: You need [Ollama](https://ollama.com/) installed and running locally.
   - Pull the model used in the code (default is `qwen3:8b`, configurable):
     ```bash
     ollama pull qwen3:8b
     ```
     _(Note: check `rag_test.py` to confirm the exact model name configured)_

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The workflow consists of three steps: preparing data, generating embeddings, and chatting.

### 1. Prepare Data

Place your PDF file in the `sources/` directory. By default, the project looks for `postgresql-basics.pdf`.

To change the target file, edit `constants.py`:

```python
DATA_DOC_NAME = 'your-file-name-without-extension'
```

Run the chunker:

```bash
python make_chunks.py
```

This creates a `.chunks.jsonl` file in the `data/` directory.

### 2. Generate Embeddings

Create vector embeddings for your chunks:

```bash
python embed_chunks.py
```

This saves `.npy` (embeddings matrix) and `.embeddings.meta.jsonl` files in `data/`.

### 3. Chat (RAG)

Start the chat interface:

```bash
python rag_test.py
```

You can now ask questions about your document. Type `exit` to quit.

## Project Structure

- `make_chunks.py`: Extracts text from PDF, cleans it, and splits it into semantic chunks.
- `embed_chunks.py`: Generates vector embeddings for the chunks using `sentence-transformers`.
- `rag_test.py`: The main CLI application that retrieves relevant chunks and queries the LLM.
- `constants.py`: Configuration file for the document name.
- `data/`: Stores processed chunks and embeddings.
- `sources/`: Directory for input PDF files.
- `utils/`: Helper utilities for text processing.

## Configuration

- **LLM Model**: To change the LLM model (e.g., to Llama 3), edit `rag_test.py` in the `call_llm` function.
- **Embedding Model**: Default is `sentence-transformers/all-MiniLM-L6-v2`. Change `MODEL_NAME_DENSE` in `embed_chunks.py` and `rag_test.py` to use a different one.
