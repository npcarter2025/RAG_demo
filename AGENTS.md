# Repository Overview

## Project Description

**RootSearch** is a sophisticated Retrieval Augmented Generation (RAG) CLI system designed for chatting with documents and code repositories. The project provides three progressively enhanced implementations:

- **Purpose**: Enable intelligent document and code search using local or cloud LLMs with semantic understanding
- **Main Goal**: Create a flexible, feature-rich RAG system that can understand code structure, maintain conversation history, and provide precise, context-aware answers
- **Key Technologies**: 
  - LangChain for RAG orchestration
  - ChromaDB for vector storage
  - Ollama (local) or OpenAI for LLM inference
  - HuggingFace sentence-transformers for embeddings
  - Python AST parsing for code understanding

## Architecture Overview

### High-Level Architecture

```
User Input → Document Loader → Chunking → Embeddings → Vector Store
                                                            ↓
User Query → Question Embedding → Similarity Search → Context Retrieval
                                                            ↓
                                        Prompt Construction → LLM → Response
```

### Main Components

1. **Document Ingestion Pipeline**
   - File discovery and filtering (.txt, .md, .py)
   - Content loading with encoding handling
   - Smart chunking (AST-based for Python, text-based for others)
   - Metadata extraction (file paths, function names, line numbers)

2. **Vector Store & Indexing**
   - ChromaDB for persistent vector storage
   - HuggingFace embeddings (all-MiniLM-L6-v2)
   - Incremental indexing with file hash tracking
   - Collection/namespace support for multiple projects

3. **Retrieval System**
   - Semantic search via vector similarity
   - Metadata filtering (by file, function, class)
   - Top-k retrieval with configurable k
   - Source document tracking

4. **LLM Integration**
   - Dual mode: Ollama (local) or OpenAI (cloud)
   - Conversational retrieval chain with memory
   - Retry logic for fault tolerance
   - Streaming support for code generation

5. **Interactive Chat Interface**
   - CLI-based conversation interface
   - Persistent conversation history
   - Code extraction and file creation
   - Inline editing with diff preview (Inline_Edit_Rag.py)

### Data Flow

1. **Indexing Flow**: Documents → Chunking → Embeddings → ChromaDB storage
2. **Query Flow**: User question → Embedding → Vector search → LLM augmentation → Response
3. **Edit Flow** (Inline_Edit_Rag.py): Instruction → LLM generation → Diff preview → User approval → File write → Reindex

## Directory Structure

### Core Files

- **`simple_rag.py`** - Basic RAG implementation with core functionality
  - Document loading (txt, md, py)
  - Basic chunking (RecursiveCharacterTextSplitter)
  - Simple QA chain with memory
  - Code block extraction and file creation

- **`Dynamic_Rag.py`** - Enhanced version with advanced features
  - Persistent conversation memory (`.rag_memory.json`)
  - Incremental indexing (`.file_hashes.json`)
  - AST-based Python parsing for function/class-level chunks
  - Metadata filtering (by file, function, class)
  - Collection/namespace support
  - Retry logic and better error handling

- **`Inline_Edit_Rag.py`** - Full-featured version with inline editing
  - All Dynamic_Rag.py features
  - Inline file editing with LLM assistance
  - Diff preview before applying changes
  - Backup system (`.backups/` directory)
  - Auto-reindexing after edits
  - User approval workflow

### Supporting Files

- **`requirements.txt`** - Python dependencies
- **`RAG_GAMEPLAN.md`** - Comprehensive technical planning document
- **`DYNAMIC_RAG_IMPROVEMENTS.md`** - Feature documentation and comparison
- **`MODEL_RECOMMENDATIONS.md`** - Ollama model recommendations
- **`documents/`** - Default directory for documents to index
- **`chroma_db/`** - Persistent vector store (auto-created)
- **`.backups/`** - Backup files before editing (auto-created)
- **`.rag_memory.json`** - Persistent conversation history (auto-created)
- **`.file_hashes.json`** - File change tracking for incremental indexing (auto-created)

### Test Files

- `test_add_numbers.py` - Unit tests for sample code
- `test_edit_command.py` - Tests for edit functionality
- `test_inline_edit.py` - Inline edit feature tests
- `test_inline_edit_standalone.py` - Standalone edit tests

## Development Workflow

### Initial Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up Ollama (for local LLM)
# Install from https://ollama.ai
ollama pull gemma3:1b  # or llama3.1:latest, mistral, etc.
ollama serve

# 3. (Optional) Set up OpenAI
# Create .env file with:
# OPENAI_API_KEY=your_key_here

# 4. Create documents directory
mkdir documents
# Add your .txt, .md, or .py files
```

### Running the System

```bash
# Basic usage (simple_rag.py)
python simple_rag.py

# Enhanced version with incremental indexing
python Dynamic_Rag.py --documents ./documents

# Full-featured with inline editing
python Inline_Edit_Rag.py --documents ./documents

# Use different model
python Dynamic_Rag.py --model llama3.1:latest

# Use OpenAI instead
python Dynamic_Rag.py --openai

# Force full reindex
python Dynamic_Rag.py --reindex

# Use separate collection
python Dynamic_Rag.py --collection myproject
```

### Interactive Commands

While in chat:
- Type questions naturally
- `quit` or `exit` - End conversation
- `clear` - Clear conversation history
- `filter: filename.py` - Filter by file
- `filter: function name` - Filter by function
- `filter: clear` - Clear filter
- `edit: filename.py instruction` - Edit file (Inline_Edit_Rag.py only)

### Building and Testing

```bash
# Run tests (if pytest is installed)
pytest test_*.py

# Test specific RAG version
python simple_rag.py --documents ./documents
python Dynamic_Rag.py --documents ./documents
python Inline_Edit_Rag.py --documents ./documents

# Test with single file
python Dynamic_Rag.py --documents ./specific_file.py
```

### Code Style

- Python 3.x (f-strings, type hints where applicable)
- UTF-8 encoding for all files
- Docstrings for classes and major functions
- Clear variable naming (snake_case)
- Error handling with try/except blocks
- User-friendly output with emojis (✅, ❌, 🔄, etc.)

### Development Best Practices

1. **Always backup before editing** - Inline_Edit_Rag.py does this automatically
2. **Test with small repos first** - Verify functionality before large-scale indexing
3. **Use incremental indexing** - Much faster for iterative development
4. **Check file hashes** - `.file_hashes.json` shows what's indexed
5. **Monitor memory usage** - Large repos can consume significant RAM
6. **Use collections for organization** - Separate projects into different collections

### Key Features to Test

- ✅ Document loading (txt, md, py)
- ✅ Chunking (text-based and AST-based)
- ✅ Vector search and retrieval
- ✅ Conversation memory persistence
- ✅ Incremental indexing
- ✅ Metadata filtering
- ✅ Code generation and extraction
- ✅ Inline editing with diff preview
- ✅ Backup and restore
- ✅ Multiple LLM backends (Ollama, OpenAI)

### Common Development Tasks

**Add support for new file types:**
1. Update `supported_extensions` in `load_documents()`
2. Add file type detection logic
3. Implement appropriate chunking strategy
4. Test with sample files

**Improve chunking strategy:**
1. Modify `parse_python_file()` or create new parser
2. Adjust chunk size/overlap in `RecursiveCharacterTextSplitter`
3. Add language-specific AST parsing
4. Test retrieval quality

**Enhance metadata:**
1. Add new fields to metadata dict in chunking functions
2. Update retrieval to use new metadata
3. Add filter commands in chat interface
4. Test filtering functionality

**Optimize performance:**
1. Adjust `k` parameter in retriever (number of chunks)
2. Tune chunk size and overlap
3. Use different embedding models
4. Implement caching strategies

### Troubleshooting

**"No files found"**: Check documents directory path and file extensions

**"Ollama error"**: Ensure `ollama serve` is running and model is pulled

**"Slow responses"**: Try smaller model (gemma3:1b) or reduce k value

**"Out of memory"**: Index smaller directories or increase system RAM

**"Bad retrievals"**: Adjust chunk size, overlap, or try different embedding model

**"Edit failed"**: Check file permissions and backup directory

## Technology Stack Summary

- **Language**: Python 3.x
- **RAG Framework**: LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace sentence-transformers
- **LLM**: Ollama (local) or OpenAI (cloud)
- **Code Parsing**: Python AST module
- **Dependencies**: See requirements.txt
- **Storage**: Local filesystem (documents, backups, vector DB)
