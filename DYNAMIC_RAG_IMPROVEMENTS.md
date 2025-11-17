# Dynamic RAG Improvements Documentation

## Overview

This document explains all the enhancements made to create `Dynamic_Rag.py` from `simple_rag.py`. The new version includes advanced features for better performance, usability, and code understanding.

---

## Table of Contents

1. [Persistent Conversation Memory](#1-persistent-conversation-memory)
2. [Incremental Indexing](#2-incremental-indexing)
3. [Metadata Filtering](#3-metadata-filtering)
4. [Code-Specific Chunking (AST Parsing)](#4-code-specific-chunking-ast-parsing)
5. [Collections/Namespaces Support](#5-collectionsnamespaces-support)
6. [Better Error Handling](#6-better-error-handling)
7. [Enhanced Source Tracking](#7-enhanced-source-tracking)
8. [Usage Examples](#usage-examples)

---

## 1. Persistent Conversation Memory

### What Was Added

- **File**: `.rag_memory.json` - Stores conversation history on disk
- **Methods**: `load_memory()`, `save_memory()`
- **Auto-save**: Conversation history is saved after each exchange

### Why It's Useful

**Before**: Conversation history was lost when you quit the program.

**After**: 
- Conversation history persists across sessions
- You can resume conversations
- History is searchable (if you add that feature later)

### How It Works

```python
# On startup
self.load_memory()  # Loads previous conversations from .rag_memory.json

# After each question/answer
self.save_memory()  # Saves current conversation to disk

# On exit
self.save_memory()  # Final save before quitting
```

### File Format

```json
{
  "messages": [
    {"type": "human", "content": "What does this function do?"},
    {"type": "ai", "content": "The function rotates a matrix..."}
  ],
  "last_updated": "2025-11-16T19:10:50.254816"
}
```

---

## 2. Incremental Indexing

### What Was Added

- **File**: `.file_hashes.json` - Tracks MD5 hashes of all indexed files
- **Method**: `get_file_hash()` - Calculates file hash
- **Smart Detection**: Only reindexes files that have changed

### Why It's Useful

**Before**: Had to reindex everything with `--reindex`, even if only one file changed.

**After**:
- Only new/changed files are indexed
- Much faster updates
- Automatically detects file changes

### How It Works

1. **First Run**: Indexes all files, saves their hashes
2. **Subsequent Runs**: 
   - Calculates hash of each file
   - Compares with stored hash
   - Only processes files with different hashes
3. **File Changes**: If you edit a file, its hash changes → gets reindexed

### Example Output

```
🔄 Checking for new or changed files...
📄 New: new_file.py          # New file detected
🔄 Updated: modified.py      # Changed file detected
⏭️  Skipped (unchanged): old_file.py  # No changes
```

### File Format

```json
{
  "testing.txt": "fdca20efaf74cbeaf80f9db409b541e5",
  "can_you_write_a_python_program.py": "03481ed9b3c3480206bb0fac801423dd"
}
```

---

## 3. Metadata Filtering

### What Was Added

- **Rich Metadata**: File paths, function names, class names, line numbers, language
- **Filter Commands**: `filter: filename.py`, `filter: function name`
- **Filtered Retrieval**: Search only within specific files or functions

### Why It's Useful

**Before**: Could only search everything at once.

**After**:
- "Show me functions in utils.py"
- "What does process_data do?" (filtered to that function)
- Better precision in answers

### Metadata Structure

Each chunk now includes:
```python
{
    "source": "/path/to/file.py",
    "type": "function",           # or "class", "text", "module_docstring"
    "name": "rotate_matrix",     # Function/class name
    "language": "python",
    "file_name": "file.py",
    "line_start": 1,
    "line_end": 24
}
```

### How to Use

```
You: filter: utils.py
✅ Filter set to: utils.py

You: What functions are here?
🤖: [Only searches in utils.py]

You: filter: function rotate_matrix
✅ Filter set to function: rotate_matrix

You: How does this work?
🤖: [Only searches rotate_matrix function]

You: filter: clear
✅ Filter cleared
```

---

## 4. Code-Specific Chunking (AST Parsing)

### What Was Added

- **AST Parser**: `parse_python_file()` - Parses Python files using Abstract Syntax Tree
- **Function Extraction**: Each function becomes its own chunk
- **Class Extraction**: Each class becomes its own chunk
- **Structure Preservation**: Code structure is maintained

### Why It's Useful

**Before**: Fixed-size chunks (1000 chars) could break functions in half.

**After**:
- Functions stay intact
- Better code understanding
- Can answer "where is function X used?"
- Preserves code context

### How It Works

1. **Parse Python File**: Uses Python's `ast` module
2. **Extract Functions**: Finds all `def` statements
3. **Extract Classes**: Finds all `class` statements
4. **Extract Docstrings**: Gets function/class documentation
5. **Create Chunks**: Each function/class becomes a Document with metadata

### Example

**Input File** (`example.py`):
```python
def rotate_matrix(matrix):
    """Rotates matrix 90 degrees."""
    # ... code ...

class MatrixProcessor:
    """Processes matrices."""
    # ... code ...
```

**Output Chunks**:
- Chunk 1: `rotate_matrix` function with metadata `{"type": "function", "name": "rotate_matrix"}`
- Chunk 2: `MatrixProcessor` class with metadata `{"type": "class", "name": "MatrixProcessor"}`

### Fallback

If AST parsing fails (syntax errors, etc.), falls back to regular text chunking.

---

## 5. Collections/Namespaces Support

### What Was Added

- **Collection Parameter**: `--collection` flag
- **Separate Indexes**: Each collection is a separate ChromaDB collection
- **Isolation**: Different projects don't mix

### Why It's Useful

**Before**: One big index for everything.

**After**:
- Separate indexes per project
- Can work on multiple codebases
- Better organization

### How to Use

```bash
# Project 1
python Dynamic_Rag.py --documents project1/ --collection project1

# Project 2
python Dynamic_Rag.py --documents project2/ --collection project2

# Default
python Dynamic_Rag.py --documents . --collection default
```

### How It Works

ChromaDB collections are like separate databases. Each collection name creates a separate index, so you can have:
- `default` collection: General documents
- `project1` collection: Project 1 codebase
- `project2` collection: Project 2 codebase

---

## 6. Better Error Handling

### What Was Added

- **Retry Logic**: 3 attempts by default
- **Better Error Messages**: More helpful error descriptions
- **Graceful Fallbacks**: Falls back to regular chunking if AST fails

### Why It's Useful

**Before**: One error = complete failure.

**After**:
- Retries on transient errors
- Better error messages help debugging
- System is more robust

### Retry Logic

```python
for attempt in range(3):  # Try 3 times
    try:
        result = self.qa_chain.invoke({"question": question})
        return result
    except Exception as e:
        if attempt < 2:
            print(f"⚠️  Attempt {attempt + 1} failed, retrying...")
            continue
        else:
            return f"❌ Error after 3 attempts: {e}"
```

---

## 7. Enhanced Source Tracking

### What Was Added

- **Rich Source Info**: Shows file name, function name, type
- **Better Formatting**: `filename.py (function: function_name)`
- **Metadata Display**: More context about where answers come from

### Why It's Useful

**Before**: Just showed file names.

**After**:
- Know exactly which function/class provided the answer
- Better traceability
- Easier to verify answers

### Example Output

**Before**:
```
📄 Sources: example.py
```

**After**:
```
📄 Sources: example.py (function: rotate_matrix), utils.py (class: MatrixProcessor)
```

---

## Usage Examples

### Basic Usage

```bash
# Index documents with incremental support
python Dynamic_Rag.py --documents .

# Force full reindex
python Dynamic_Rag.py --documents . --reindex

# Disable incremental indexing
python Dynamic_Rag.py --documents . --no-incremental

# Use different collection
python Dynamic_Rag.py --documents . --collection myproject
```

### In Chat Interface

```
You: filter: utils.py
✅ Filter set to: utils.py

You: What functions are in this file?
🤖: [Lists functions in utils.py]

You: filter: function process_data
✅ Filter set to function: process_data

You: How does this work?
🤖: [Explains process_data function]

You: filter: clear
✅ Filter cleared

You: What's in the codebase?
🤖: [Searches all files]
```

### Code Generation

```
You: Write a binary tree traversal

🤖: [Shows code with code blocks]

💡 Found python code block. What would you like to do?
   [s] Save to file
   [d] Display code only (don't save)
   [n] Nothing (skip)

Your choice (s/d/n): s

✅ Created Python file: write_a_binary_tree_traversal.py
```

---

## File Structure

### New Files Created

- `Dynamic_Rag.py` - Enhanced RAG system
- `.rag_memory.json` - Conversation history (auto-created)
- `.file_hashes.json` - File hash tracking (auto-created)

### Persistence Files

These files are automatically created and managed:
- **`.rag_memory.json`**: Conversation history
- **`.file_hashes.json`**: File change tracking
- **`chroma_db/`**: Vector database (existing)

---

## Key Differences from simple_rag.py

| Feature | simple_rag.py | Dynamic_Rag.py |
|---------|---------------|----------------|
| **Memory** | In-memory only | Persistent (saves to disk) |
| **Indexing** | Full reindex always | Incremental (only changed files) |
| **Metadata** | Basic (file paths) | Rich (functions, classes, line numbers) |
| **Chunking** | Fixed-size chunks | AST parsing for Python (function/class level) |
| **Filtering** | None | Filter by file, function, class |
| **Collections** | Single index | Multiple collections/namespaces |
| **Error Handling** | Basic | Retry logic, better messages |
| **Source Info** | File names only | File + function/class names |

---

## Technical Details

### AST Parsing Implementation

```python
def parse_python_file(self, file_path: Path, content: str) -> List[Document]:
    tree = ast.parse(content)
    
    # Extract functions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_code = ast.get_source_segment(content, node)
            # Create Document with metadata
            chunks.append(Document(
                page_content=func_code,
                metadata={
                    "type": "function",
                    "name": node.name,
                    "line_start": node.lineno,
                    ...
                }
            ))
```

### Incremental Indexing Logic

```python
file_hash = self.get_file_hash(file_path)
stored_hash = self.file_hashes.get(str(file_path))

if file_hash != stored_hash:
    # File changed, reindex it
    process_file(file_path)
    self.file_hashes[str(file_path)] = file_hash
else:
    # File unchanged, skip it
    skip_file(file_path)
```

### Metadata Filtering

```python
# Filter by file
filter_metadata = {"file_name": "utils.py"}

# Filter by function
filter_metadata = {"type": "function", "name": "process_data"}

# Apply filter to retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5, "filter": filter_metadata}
)
```

---

## Benefits Summary

1. **Faster Updates**: Only reindexes changed files
2. **Better Code Understanding**: AST parsing preserves structure
3. **Precise Queries**: Filter by file, function, or class
4. **Persistent Memory**: Conversations survive restarts
5. **Better Organization**: Collections for multiple projects
6. **More Robust**: Retry logic and better error handling
7. **Better Traceability**: Enhanced source information

---

## Future Enhancements (Not Yet Implemented)

- **Hybrid Search**: Combine vector + keyword search
- **Reranking**: Improve result relevance
- **Web Interface**: Browser-based UI
- **Query History**: Search past conversations
- **Multi-language AST**: Support JavaScript, TypeScript, etc.
- **Git Integration**: Track code changes over time
- **Dependency Tracking**: Map function/class relationships

---

## Conclusion

`Dynamic_Rag.py` is a significant upgrade from `simple_rag.py` with:
- ✅ Persistent conversation memory
- ✅ Incremental indexing
- ✅ Rich metadata and filtering
- ✅ Code-specific chunking (AST parsing)
- ✅ Collections/namespaces
- ✅ Better error handling
- ✅ Enhanced source tracking

All features are working and tested. The system is production-ready for personal use and can be extended further as needed.

