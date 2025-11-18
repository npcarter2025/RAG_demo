# RootSearch - RAG CLI System
Someone help me thing of a better name please.

A sophisticated Retrieval Augmented Generation (RAG) CLI system for chatting with documents and code repositories. Three progressively enhanced implementations are provided.

## 🚀 Features

- ✅ Chat with your documents in the terminal
- ✅ Supports `.txt`, `.md`, and `.py` files
- ✅ Supports just about Every file type at this point too.
- File support for .pdf coming soon.
- ✅ Uses Ollama (local LLM) by default -
- ✅ Optional OpenRouter support
- ✅ Optional OpenAI support
- ✅ Persistent vector store (only indexes once)
- ✅ Conversation memory (remembers context)
- ✅ Shows source documents for answers
- ✅ Incremental indexing (only reindexes changed files)
- ✅ Metadata filtering (by file, function, class)
- ✅ AST-based code parsing for Python files
- ✅ Inline code editing with diff preview (Inline_Edit_Rag.py)

## 📦 Three Versions

### 1. `simple_rag.py` - Basic RAG
- Core RAG functionality
- Document loading and chunking
- Basic Q&A with memory
- Code generation and file creation

### 2. `Dynamic_Rag.py` - Enhanced RAG
- All features from `simple_rag.py`
- **Persistent conversation memory** (saves to `.rag_memory.json`)
- **Incremental indexing** (only reindexes changed files)
- **Metadata filtering** (filter by file, function, class)
- **AST-based Python parsing** (function/class-level chunks)
- **Collections/namespaces** (multiple indexes)
- Better error handling with retries

### 3. `Inline_Edit_Rag.py` - Full-Featured RAG
- All features from `Dynamic_Rag.py`
- **Inline code editing** with diff preview
- **Automatic backups** before editing
- **User approval workflow** for changes
- **Auto-reindexing** after edits

See `DYNAMIC_RAG_IMPROVEMENTS.md` for detailed feature documentation.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Ollama (Default)

If you don't have Ollama installed:

```bash
# Install Ollama from https://ollama.ai
# Then pull a model:
ollama pull llama2
```

Make sure Ollama is running:
```bash
ollama serve
```

### 3. Add Your Documents

Create a `documents` directory and add your `.txt` and `.md` files:

```bash
mkdir documents
# Copy your files
cp your_file.txt documents/
cp your_file.md documents/
```

### 4. Run the CLI

Choose which version to use:

```bash
# Basic version
python simple_rag.py

# Enhanced version (recommended)
python Dynamic_Rag.py --documents .

# Full-featured with inline editing
python Inline_Edit_Rag.py --documents .
```

The first time, it will:
- Load your documents
- Create chunks
- Generate embeddings
- Index everything in a vector store

After that, it starts an interactive chat!

## Usage

### Basic Usage

```bash
# Use default Ollama model (llama2)
python simple_rag.py

# Use a different Ollama model
python simple_rag.py --model mistral

# Use OpenAI instead (requires OPENAI_API_KEY in .env)
python simple_rag.py --openai

# Specify a different documents directory
python simple_rag.py --documents /path/to/docs

# Force reindexing (if you added new files)
python simple_rag.py --reindex
```

### Command Line Options

- `--documents DIR`: Directory containing documents (default: `documents`)
- `--reindex`: Force reindexing of all documents
- `--openai`: Use OpenAI API instead of Ollama
- `--model NAME`: Ollama model name (default: `llama2`)

### Interactive Commands

While in the chat:
- Type your question and press Enter
- Type `quit` or `exit` to end the conversation
- Type `clear` to clear conversation history

## Examples

```
You: What is the main topic of these documents?

🤖 Assistant: Based on the documents, the main topic is...

📄 Sources: document1.txt, document2.md

You: Can you summarize the key points?

🤖 Assistant: [Summary of key points from the documents]
```

## Using OpenAI (Optional)

If you want to use OpenAI instead of Ollama:

1. Create a `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
```

2. Run with `--openai` flag:
```bash
python simple_rag.py --openai
```

## Recommended Ollama Models

- `llama2` - Good balance of quality and speed (default)
- `mistral` - Fast and efficient
- `codellama` - Good for code-related documents
- `llama2:13b` - Better quality, slower
- `mistral:7b` - Good alternative

Check available models:
```bash
ollama list
```

Pull a new model:
```bash
ollama pull mistral
```

## How It Works

1. **Document Loading**: Loads all `.txt` and `.md` files from the documents directory
2. **Chunking**: Splits documents into smaller chunks (1000 chars with 200 overlap)
3. **Embedding**: Creates embeddings using HuggingFace's `all-MiniLM-L6-v2` model
4. **Indexing**: Stores chunks and embeddings in ChromaDB (vector database)
5. **Retrieval**: When you ask a question, retrieves relevant chunks
6. **Generation**: Sends question + retrieved context to LLM (Ollama/OpenAI)
7. **Response**: Returns answer with source documents

## File Structure

```
RootSearch/
├── simple_rag.py              # Basic RAG implementation
├── Dynamic_Rag.py             # Enhanced RAG with advanced features
├── Inline_Edit_Rag.py         # Full-featured RAG with inline editing
├── requirements.txt           # Dependencies
├── documents/                 # Put your .txt, .md, and .py files here
├── chroma_db/                # Vector store (created automatically)
├── .rag_memory.json          # Conversation history (Dynamic/Inline versions)
├── .file_hashes.json         # File change tracking (Dynamic/Inline versions)
├── .backups/                 # Backup files before editing (Inline version)
├── DYNAMIC_RAG_IMPROVEMENTS.md  # Feature documentation
├── MODEL_RECOMMENDATIONS.md   # Model selection guide
├── RAG_GAMEPLAN.md           # Technical planning document
└── README.md                  # This file
```

## Troubleshooting

### "No .txt or .md files found"
- Make sure you have a `documents` directory
- Add some `.txt` or `.md` files to it

### "Error connecting to Ollama"
- Make sure Ollama is running: `ollama serve`
- Check that your model is available: `ollama list`
- Pull the model if needed: `ollama pull llama2`

### "OPENAI_API_KEY not found"
- Create a `.env` file with your API key
- Or use Ollama (default) instead

### Slow responses
- Ollama models run locally and can be slower than cloud APIs
- Try a smaller model: `--model mistral`
- Or use OpenAI for faster responses

## Next Steps

- Add more document types (PDF, Word, etc.)
- Improve chunking strategy
- Add web interface
- Support multiple repositories
- Add query history

## License

MIT





