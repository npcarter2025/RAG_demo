#!/usr/bin/env python3
"""
Inline Edit RAG CLI - RAG system with inline code editing capabilities
Features:
- All features from Dynamic_Rag.py
- Inline editing of existing files (diff-based)
- Backup before editing
- Preview changes with diff
- User approval workflow
- Auto-reindex after edits
"""

import os
import sys
import re
import json
import hashlib
import ast
import difflib
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class InlineEditRAG:
    def __init__(
        self,
        documents_path: str = "documents",
        use_openai: bool = False,
        ollama_model: str = "gemma3:1b",
        collection_name: str = "default",
        memory_file: str = ".rag_memory.json"
    ):
        """
        Initialize the RAG system with inline editing capabilities.
        
        Args:
            documents_path: Path to a file or directory containing .txt, .md, or .py files
            use_openai: If True, use OpenAI API. If False, use local LLM (Ollama)
            ollama_model: Ollama model name (default: "gemma3:1b")
            collection_name: Name of the ChromaDB collection (allows multiple indexes)
            memory_file: Path to save conversation history
        """
        self.documents_path = Path(documents_path)
        self.use_openai = use_openai
        self.ollama_model = ollama_model
        self.collection_name = collection_name
        self.memory_file = Path(memory_file)
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.file_hashes = {}  # Track file hashes for incremental indexing
        self.hash_file = Path(".file_hashes.json")
        self.backup_dir = Path(".backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Create documents directory if it doesn't exist (only if it's a directory)
        if self.documents_path.is_dir() or not self.documents_path.exists():
            self.documents_path.mkdir(exist_ok=True)
        
        # Load conversation history
        self.load_memory()
        
        # Load file hashes for incremental indexing
        self.load_file_hashes()
        
        # Initialize embeddings (free, local)
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    
    def load_memory(self):
        """Load conversation history from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    # Restore conversation history
                    for msg in history.get('messages', []):
                        if msg['type'] == 'human':
                            self.memory.chat_memory.add_user_message(msg['content'])
                        elif msg['type'] == 'ai':
                            self.memory.chat_memory.add_ai_message(msg['content'])
                print(f"✅ Loaded conversation history from {self.memory_file}")
            except Exception as e:
                print(f"⚠️  Could not load conversation history: {e}")
    
    def save_memory(self):
        """Save conversation history to disk."""
        try:
            history = {'messages': [], 'last_updated': datetime.now().isoformat()}
            if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                for msg in self.memory.chat_memory.messages:
                    msg_type = 'human' if hasattr(msg, 'content') and msg.__class__.__name__ == 'HumanMessage' else 'ai'
                    history['messages'].append({
                        'type': msg_type,
                        'content': msg.content if hasattr(msg, 'content') else str(msg)
                    })
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save conversation history: {e}")
    
    def load_file_hashes(self):
        """Load file hashes for incremental indexing."""
        if self.hash_file.exists():
            try:
                with open(self.hash_file, 'r', encoding='utf-8') as f:
                    self.file_hashes = json.load(f)
            except Exception as e:
                print(f"⚠️  Could not load file hashes: {e}")
                self.file_hashes = {}
    
    def save_file_hashes(self):
        """Save file hashes for incremental indexing."""
        try:
            with open(self.hash_file, 'w', encoding='utf-8') as f:
                json.dump(self.file_hashes, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save file hashes: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate hash of a file for change detection."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            return ""
    
    def backup_file(self, file_path: Path) -> Optional[Path]:
        """Create a backup of a file before editing."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = self.backup_dir / backup_name
            
            shutil.copy2(file_path, backup_path)
            return backup_path
        except Exception as e:
            print(f"⚠️  Could not create backup: {e}")
            return None
    
    def create_diff(self, old_content: str, new_content: str, filename: str = "file") -> str:
        """Create a unified diff between old and new content."""
        # Normalize line endings and ensure both end with newline
        old_content = old_content.replace('\r\n', '\n').replace('\r', '\n')
        new_content = new_content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split into lines, preserving newlines
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        # If content doesn't end with newline, add it for proper diff
        if old_lines and not old_lines[-1].endswith('\n'):
            old_lines[-1] += '\n'
        if new_lines and not new_lines[-1].endswith('\n'):
            new_lines[-1] += '\n'
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=filename,
            tofile=filename,
            lineterm='',
            n=3
        )
        
        return ''.join(diff)
    
    def find_function_or_class(self, file_path: Path, name: str) -> Optional[Dict]:
        """
        Find a function or class by name in a Python file.
        Returns dict with 'type', 'name', 'start_line', 'end_line', 'code', 'full_content'.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == name:
                        code = ast.get_source_segment(content, node)
                        return {
                            'type': 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class',
                            'name': node.name,
                            'start_line': node.lineno,
                            'end_line': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                            'code': code or '',
                            'full_content': content
                        }
            
            return None
        except Exception as e:
            print(f"⚠️  Error parsing {file_path}: {e}")
            return None
    
    def replace_function_or_class(self, file_path: Path, name: str, new_code: str) -> Tuple[bool, str, Optional[str]]:
        """
        Replace a function or class in a file with new code.
        Returns (success, message, diff).
        """
        # Find the function/class
        func_info = self.find_function_or_class(file_path, name)
        if not func_info:
            return False, f"❌ Could not find function/class '{name}' in {file_path}", None
        
        # Read original file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            return False, f"❌ Error reading file: {e}", None
        
        # Create backup
        backup_path = self.backup_file(file_path)
        if not backup_path:
            return False, "❌ Could not create backup. Aborting edit.", None
        
        # Replace the function/class code
        lines = original_content.splitlines(keepends=True)
        start_idx = func_info['start_line'] - 1
        end_idx = func_info['end_line']
        
        # Extract the new code (remove function/class definition if it's in the new_code)
        new_code_lines = new_code.splitlines(keepends=True)
        
        # Build new content
        new_lines = lines[:start_idx] + new_code_lines + lines[end_idx:]
        new_content = ''.join(new_lines)
        
        # Create diff
        diff = self.create_diff(original_content, new_content, file_path.name)
        
        # Show preview
        print(f"\n📝 Proposed changes to {file_path.name}:")
        print("=" * 60)
        print(diff)
        print("=" * 60)
        
        # Ask for approval
        while True:
            choice = input("\nApply changes? [y/n]: ").strip().lower()
            if choice == 'y' or choice == 'yes':
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    # Update file hash
                    self.file_hashes[str(file_path)] = self.get_file_hash(file_path)
                    self.save_file_hashes()
                    
                    return True, f"✅ Updated {file_path.name} (backup: {backup_path.name})", diff
                except Exception as e:
                    return False, f"❌ Error writing file: {e}", None
            elif choice == 'n' or choice == 'no':
                return False, "❌ Edit cancelled by user.", diff
            else:
                print("Please enter 'y' or 'n'")
    
    def edit_file_inline(self, file_path: Path, instruction: str) -> Tuple[bool, str]:
        """
        Edit a file based on a natural language instruction.
        Uses the LLM to generate the edit, then applies it with diff preview.
        """
        if not self.qa_chain:
            return False, "❌ QA chain not set up. Please index documents first."
        
        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            return False, f"❌ Error reading file: {e}"
        
        # Ask LLM to generate the edited version
        # For small models, use a more structured prompt
        prompt = f"""You are a code editor. Edit the following Python file according to the instruction.

CURRENT FILE ({file_path.name}):
```python
{file_content}
```

INSTRUCTION: {instruction}

REQUIREMENTS:
1. You MUST modify the code according to the instruction
2. Return the COMPLETE file with your changes
3. Preserve all existing code that doesn't need to change
4. Only modify what the instruction asks for
5. Keep the same file structure, imports, and formatting

Return ONLY the complete edited code in a markdown code block:
```python
[complete file with edits applied]
```

Do not include explanations. Only return the code block."""
        
        try:
            result = self.qa_chain.invoke({"question": prompt})
            answer = result["answer"]
            
            # Extract code block
            code_blocks = self.extract_code_blocks(answer)
            if not code_blocks:
                return False, "❌ LLM did not return code in a code block. Try rephrasing your request."
            
            # Get the Python code (should be the full file)
            new_content = code_blocks[0][1]  # (lang, code)
            
            # Normalize whitespace for comparison
            old_normalized = file_content.replace('\r\n', '\n').replace('\r', '\n').strip()
            new_normalized = new_content.replace('\r\n', '\n').replace('\r', '\n').strip()
            
            # Check if content actually changed
            if old_normalized == new_normalized:
                return False, "❌ LLM returned unchanged code. The edit instruction may not have been clear enough, or the model didn't make the requested changes. Try rephrasing your instruction."
            
            # Create backup
            backup_path = self.backup_file(file_path)
            if not backup_path:
                return False, "❌ Could not create backup. Aborting edit."
            
            # Create diff
            diff = self.create_diff(file_content, new_content, file_path.name)
            
            # Show preview
            print(f"\n📝 Proposed changes to {file_path.name}:")
            print("=" * 60)
            print(diff)
            print("=" * 60)
            
            # Ask for approval
            while True:
                choice = input("\nApply changes? [y/n]: ").strip().lower()
                if choice == 'y' or choice == 'yes':
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        
                        # Update file hash
                        self.file_hashes[str(file_path)] = self.get_file_hash(file_path)
                        self.save_file_hashes()
                        
                        return True, f"✅ Updated {file_path.name} (backup: {backup_path.name})"
                    except Exception as e:
                        return False, f"❌ Error writing file: {e}"
                elif choice == 'n' or choice == 'no':
                    return False, "❌ Edit cancelled by user."
                else:
                    print("Please enter 'y' or 'n'")
        
        except Exception as e:
            return False, f"❌ Error generating edit: {e}"
    
    def parse_python_file(self, file_path: Path, content: str) -> List[Document]:
        """
        Parse Python file using AST to extract functions and classes.
        Returns list of Document objects with metadata.
        """
        chunks = []
        try:
            tree = ast.parse(content)
            
            # Extract file-level docstring
            docstring = ast.get_docstring(tree)
            if docstring:
                chunks.append(Document(
                    page_content=f"# {file_path.name}\n\n{docstring}",
                    metadata={
                        "source": str(file_path),
                        "type": "module_docstring",
                        "language": "python",
                        "file_name": file_path.name
                    }
                ))
            
            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_code = ast.get_source_segment(content, node) or ""
                    func_doc = ast.get_docstring(node) or ""
                    
                    chunks.append(Document(
                        page_content=f"def {node.name}:\n{func_doc}\n\n{func_code}",
                        metadata={
                            "source": str(file_path),
                            "type": "function",
                            "name": node.name,
                            "language": "python",
                            "file_name": file_path.name,
                            "line_start": node.lineno,
                            "line_end": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
                        }
                    ))
                
                elif isinstance(node, ast.ClassDef):
                    class_code = ast.get_source_segment(content, node) or ""
                    class_doc = ast.get_docstring(node) or ""
                    
                    chunks.append(Document(
                        page_content=f"class {node.name}:\n{class_doc}\n\n{class_code}",
                        metadata={
                            "source": str(file_path),
                            "type": "class",
                            "name": node.name,
                            "language": "python",
                            "file_name": file_path.name,
                            "line_start": node.lineno,
                            "line_end": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
                        }
                    ))
            
            # If no functions/classes found, chunk by size
            if not chunks:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                text_chunks = text_splitter.split_text(content)
                for i, chunk in enumerate(text_chunks):
                    chunks.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": str(file_path),
                            "type": "text",
                            "language": "python",
                            "file_name": file_path.name,
                            "chunk_index": i
                        }
                    ))
            
        except SyntaxError:
            # If AST parsing fails, fall back to regular chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            text_chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(text_chunks):
                chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": str(file_path),
                        "type": "text",
                        "language": "python",
                        "file_name": file_path.name,
                        "chunk_index": i
                    }
                ))
        
        return chunks
    
    def load_documents(self, incremental: bool = True) -> Tuple[List[Document], List[str]]:
        """
        Load .txt, .md, and .py files with metadata and incremental indexing support.
        Returns tuple of (documents, file_paths).
        """
        documents = []
        file_paths = []
        supported_extensions = ['.txt', '.md', '.py']
        new_files = []
        changed_files = []
        
        # Check if it's a single file
        if self.documents_path.is_file():
            file_path = self.documents_path
            if file_path.suffix.lower() in supported_extensions:
                file_hash = self.get_file_hash(file_path)
                stored_hash = self.file_hashes.get(str(file_path))
                
                if not incremental or file_hash != stored_hash:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            if content.strip():
                                if file_path.suffix.lower() == '.py':
                                    # Use AST parsing for Python files
                                    chunks = self.parse_python_file(file_path, content)
                                    documents.extend(chunks)
                                else:
                                    # Regular chunking for text files
                                    text_splitter = RecursiveCharacterTextSplitter(
                                        chunk_size=1000,
                                        chunk_overlap=200,
                                        length_function=len,
                                    )
                                    text_chunks = text_splitter.split_text(content)
                                    for i, chunk in enumerate(text_chunks):
                                        documents.append(Document(
                                            page_content=chunk,
                                            metadata={
                                                "source": str(file_path),
                                                "type": "text",
                                                "file_name": file_path.name,
                                                "chunk_index": i
                                            }
                                        ))
                                
                                file_paths.append(str(file_path))
                                self.file_hashes[str(file_path)] = file_hash
                                
                                if stored_hash is None:
                                    new_files.append(file_path)
                                    print(f"📄 New: {file_path}")
                                else:
                                    changed_files.append(file_path)
                                    print(f"🔄 Updated: {file_path}")
                                print(f"   Loaded: {file_path}")
                            else:
                                print(f"⚠️  File '{file_path}' is empty")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                else:
                    print(f"⏭️  Skipped (unchanged): {file_path}")
            else:
                print(f"⚠️  File '{file_path}' is not a supported file type (.txt, .md, or .py)")
        
        # Otherwise, treat it as a directory
        elif self.documents_path.is_dir():
            # Find all .txt, .md, and .py files
            for ext in ["*.txt", "*.md", "*.py"]:
                for file_path in self.documents_path.rglob(ext):
                    file_hash = self.get_file_hash(file_path)
                    stored_hash = self.file_hashes.get(str(file_path))
                    
                    if not incremental or file_hash != stored_hash:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                if content.strip():
                                    if file_path.suffix.lower() == '.py':
                                        # Use AST parsing for Python files
                                        chunks = self.parse_python_file(file_path, content)
                                        documents.extend(chunks)
                                    else:
                                        # Regular chunking for text files
                                        text_splitter = RecursiveCharacterTextSplitter(
                                            chunk_size=1000,
                                            chunk_overlap=200,
                                            length_function=len,
                                        )
                                        text_chunks = text_splitter.split_text(content)
                                        for i, chunk in enumerate(text_chunks):
                                            documents.append(Document(
                                                page_content=chunk,
                                                metadata={
                                                    "source": str(file_path),
                                                    "type": "text",
                                                    "file_name": file_path.name,
                                                    "chunk_index": i
                                                }
                                            ))
                                    
                                    file_paths.append(str(file_path))
                                    self.file_hashes[str(file_path)] = file_hash
                                    
                                    if stored_hash is None:
                                        new_files.append(file_path)
                                        print(f"📄 New: {file_path}")
                                    else:
                                        changed_files.append(file_path)
                                        print(f"🔄 Updated: {file_path}")
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
                    else:
                        print(f"⏭️  Skipped (unchanged): {file_path}")
        else:
            print(f"⚠️  Path '{self.documents_path}' does not exist")
            return [], []
        
        if not documents:
            print(f"\n⚠️  No new or changed files found at '{self.documents_path}'")
            if incremental:
                print(f"   (Use --reindex to force full reindex)")
            return [], []
        
        print(f"\n✅ Loaded {len(documents)} chunk(s) from {len(file_paths)} file(s)")
        if new_files:
            print(f"   📄 {len(new_files)} new file(s)")
        if changed_files:
            print(f"   🔄 {len(changed_files)} updated file(s)")
        
        # Save updated hashes
        self.save_file_hashes()
        
        return documents, file_paths
    
    def index_documents(self, force_reindex: bool = False, incremental: bool = True):
        """Index documents into the vector store with incremental support."""
        persist_directory = "./chroma_db"
        
        # Check if vectorstore already exists
        if not force_reindex and Path(persist_directory).exists():
            print(f"Loading existing vector store from {persist_directory}...")
            try:
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                print("✅ Loaded existing index")
                
                # Do incremental update if enabled
                if incremental:
                    print("\n🔄 Checking for new or changed files...")
                    new_docs, _ = self.load_documents(incremental=True)
                    if new_docs:
                        print(f"Adding {len(new_docs)} new/updated chunks...")
                        self.vectorstore.add_documents(new_docs)
                        print("✅ Incremental update complete")
                    else:
                        print("✅ All files up to date")
                
                return
            except Exception as e:
                print(f"Error loading existing index: {e}")
                print("Creating new index...")
        
        # Load and process documents
        documents, _ = self.load_documents(incremental=False if force_reindex else incremental)
        if not documents:
            return
        
        # Create vector store
        print("Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name=self.collection_name
        )
        print(f"✅ Indexed documents in {persist_directory} (collection: {self.collection_name})")
    
    def setup_qa_chain(self, retry_count: int = 3):
        """Set up the question-answering chain with retry logic."""
        if not self.vectorstore:
            print("❌ Vector store not initialized. Please index documents first.")
            return
        
        # Initialize LLM
        if self.use_openai:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("❌ OPENAI_API_KEY not found in environment variables.")
                print("   Please create a .env file with: OPENAI_API_KEY=your_key")
                print("   Or use Ollama (default) by removing --openai flag")
                return
            print(f"Using OpenAI: gpt-3.5-turbo")
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                api_key=api_key
            )
        else:
            # Use Ollama for local LLM
            try:
                print(f"Using Ollama model: {self.ollama_model}")
                print("   (Make sure Ollama is running: 'ollama serve')")
                llm = ChatOllama(
                    model=self.ollama_model,
                    temperature=0,
                )
            except Exception as e:
                print(f"❌ Error connecting to Ollama: {e}")
                print("   Make sure Ollama is running: 'ollama serve'")
                print(f"   And that model '{self.ollama_model}' is available: 'ollama list'")
                return
        
        # Create QA chain with metadata filtering support
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Get more results for better context
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )
        self.retry_count = retry_count
        print("✅ QA chain ready")
    
    def ask(self, question: str, filter_metadata: Optional[Dict] = None) -> str:
        """
        Ask a question and get an answer with retry logic and metadata filtering.
        
        Args:
            question: The question to ask
            filter_metadata: Optional metadata filter (e.g., {"file_name": "utils.py"})
        """
        if not self.qa_chain:
            return "❌ QA chain not set up. Please index documents first."
        
        # Apply metadata filter if provided
        if filter_metadata and self.vectorstore:
            # Create filtered retriever
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 5, "filter": filter_metadata}
            )
            # Temporarily replace retriever
            original_retriever = self.qa_chain.retriever
            self.qa_chain.retriever = retriever
        
        # Retry logic
        last_error = None
        for attempt in range(self.retry_count):
            try:
                result = self.qa_chain.invoke({"question": question})
                answer = result["answer"]
                
                # Restore original retriever if we changed it
                if filter_metadata and self.vectorstore:
                    self.qa_chain.retriever = original_retriever
                
                # Show sources with metadata
                sources = result.get("source_documents", [])
                if sources:
                    source_info = []
                    for doc in sources:
                        meta = doc.metadata if hasattr(doc, "metadata") else {}
                        source_name = meta.get("file_name", Path(meta.get("source", "")).name if meta.get("source") else "unknown")
                        source_type = meta.get("type", "text")
                        if meta.get("name"):
                            source_info.append(f"{source_name} ({source_type}: {meta['name']})")
                        else:
                            source_info.append(f"{source_name}")
                    if source_info:
                        answer += f"\n\n📄 Sources: {', '.join(set(source_info))}"
                
                return answer
            except Exception as e:
                last_error = e
                if attempt < self.retry_count - 1:
                    print(f"⚠️  Attempt {attempt + 1} failed, retrying...")
                    continue
                else:
                    error_msg = str(e)
                    if "500" in error_msg or "model runner" in error_msg.lower():
                        return f"❌ Ollama error: The model may not be loaded or there's a resource issue.\n   Try: ollama run {self.ollama_model}\n   Or check Ollama server logs.\n   Error: {error_msg}"
                    return f"❌ Error after {self.retry_count} attempts: {error_msg}"
        
        # Restore original retriever if we changed it
        if filter_metadata and self.vectorstore:
            self.qa_chain.retriever = original_retriever
        
        return f"❌ Error: {last_error}"
    
    def extract_code_blocks(self, text: str) -> List[tuple]:
        """Extract code blocks from markdown-formatted text."""
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [(lang.strip() if lang else '', code.strip()) for lang, code in matches]
    
    def create_python_file(self, filename: str, code: str, output_dir: str = ".") -> str:
        """Create a Python file with the given code."""
        try:
            if not filename.endswith('.py'):
                filename += '.py'
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            file_path = output_path / filename
            
            if file_path.exists():
                return f"⚠️  File '{file_path}' already exists. Not overwriting."
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            return f"✅ Created Python file: {file_path}"
        except Exception as e:
            return f"❌ Error creating file: {e}"
    
    def chat(self):
        """Start an interactive chat session with inline editing capabilities."""
        if not self.qa_chain:
            print("❌ QA chain not set up. Please index documents first.")
            return
        
        print("\n" + "="*60)
        print("💬 Inline Edit RAG - Enhanced Chat with Code Editing")
        print("   Type 'quit' or 'exit' to end the conversation")
        print("   Type 'clear' to clear conversation history")
        print("   Type 'filter: filename.py' to filter by file")
        print("   Type 'filter: function function_name' to filter by function")
        print("   Type 'edit: filename.py instruction' to edit a file")
        print("   Type 'edit: function function_name new_code' to edit a function")
        print("   When code is generated, you'll be asked to save or display it")
        print("="*60 + "\n")
        
        current_filter = None
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ["quit", "exit", "q"]:
                    self.save_memory()
                    print("👋 Goodbye! Conversation history saved.")
                    break
                
                if question.lower() == "clear":
                    self.memory.clear()
                    current_filter = None
                    print("✅ Conversation history cleared\n")
                    continue
                
                # Handle inline editing commands
                if question.lower().startswith("edit:"):
                    edit_cmd = question[5:].strip()
                    parts = edit_cmd.split(None, 1)
                    
                    if len(parts) < 2:
                        print("❌ Usage: edit: filename.py instruction")
                        print("   Example: edit: add_numbers.py Add error handling")
                        continue
                    
                    target = parts[0]
                    instruction = parts[1]
                    
                    # Find the file
                    file_path = None
                    if Path(target).exists():
                        file_path = Path(target)
                    elif (self.documents_path / target).exists():
                        file_path = self.documents_path / target
                    else:
                        # Search in documents directory
                        for ext in ["*.py", "*.txt", "*.md"]:
                            for f in self.documents_path.rglob(ext):
                                if f.name == target:
                                    file_path = f
                                    break
                            if file_path:
                                break
                    
                    if not file_path or not file_path.exists():
                        print(f"❌ File '{target}' not found")
                        continue
                    
                    # Perform edit
                    success, message = self.edit_file_inline(file_path, instruction)
                    print(f"\n{message}")
                    
                    if success:
                        # Reindex the file
                        print(f"\n🔄 Reindexing {file_path.name}...")
                        self.file_hashes[str(file_path)] = self.get_file_hash(file_path)
                        new_docs, _ = self.load_documents(incremental=True)
                        if new_docs and self.vectorstore:
                            self.vectorstore.add_documents(new_docs)
                            print("✅ File reindexed")
                    
                    print()
                    continue
                
                # Handle metadata filtering
                if question.lower().startswith("filter:"):
                    filter_cmd = question[7:].strip()
                    if filter_cmd.endswith('.py') or filter_cmd.endswith('.txt') or filter_cmd.endswith('.md'):
                        current_filter = {"file_name": filter_cmd}
                        print(f"✅ Filter set to: {filter_cmd}\n")
                    elif filter_cmd.startswith("function "):
                        func_name = filter_cmd[9:].strip()
                        current_filter = {"type": "function", "name": func_name}
                        print(f"✅ Filter set to function: {func_name}\n")
                    elif filter_cmd == "clear" or filter_cmd == "none":
                        current_filter = None
                        print("✅ Filter cleared\n")
                    else:
                        print("❌ Invalid filter. Use 'filter: filename.py' or 'filter: function name'\n")
                    continue
                
                print("\n🤖 Assistant: ", end="", flush=True)
                answer = self.ask(question, filter_metadata=current_filter)
                print(answer)
                
                # Save memory after each exchange
                self.save_memory()
                
                # Check if answer contains code blocks
                code_blocks = self.extract_code_blocks(answer)
                if code_blocks:
                    python_code = None
                    code_lang = None
                    for lang, code in code_blocks:
                        if lang.lower() in ['python', 'py', ''] or not lang:
                            python_code = code
                            code_lang = lang if lang else 'python'
                            break
                    
                    if not python_code and code_blocks:
                        code_lang, python_code = code_blocks[0]
                        if not code_lang:
                            code_lang = 'python'
                    
                    if python_code:
                        print(f"\n💡 Found {code_lang} code block. What would you like to do?")
                        print("   [s] Save to file")
                        print("   [d] Display code only (don't save)")
                        print("   [n] Nothing (skip)")
                        
                        while True:
                            choice = input("\nYour choice (s/d/n): ").strip().lower()
                            
                            if choice == 's' or choice == 'save':
                                filename = re.sub(r'[^\w\s-]', '', question.lower())
                                filename = re.sub(r'[-\s]+', '_', filename)
                                filename = filename[:30]
                                if not filename:
                                    filename = "generated_code"
                                filename += ".py"
                                
                                result = self.create_python_file(filename, python_code)
                                print(f"\n{result}\n")
                                break
                            
                            elif choice == 'd' or choice == 'display':
                                print(f"\n📝 Code:\n```{code_lang}")
                                print(python_code)
                                print("```\n")
                                break
                            
                            elif choice == 'n' or choice == 'nothing' or choice == '':
                                print("✅ Skipped saving code.\n")
                                break
                            
                            else:
                                print("❌ Invalid choice. Please enter 's', 'd', or 'n'.")
                
                print()
                
            except KeyboardInterrupt:
                self.save_memory()
                print("\n\n👋 Goodbye! Conversation history saved.")
                break
            except EOFError:
                self.save_memory()
                print("\n\n👋 Goodbye! Conversation history saved.")
                break


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Inline Edit RAG CLI - RAG with inline code editing capabilities"
    )
    parser.add_argument(
        "--documents",
        type=str,
        default="documents",
        help="Path to a file or directory containing .txt, .md, or .py files (default: 'documents')"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force full reindexing of all documents"
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Disable incremental indexing (reindex everything)"
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI API instead of Ollama (Ollama is the default)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:1b",
        help="Ollama model name (default: 'gemma3:1b'). Examples: gemma3:1b, llama3.1:latest, mistral, qwen2.5"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="default",
        help="ChromaDB collection name (allows multiple indexes, default: 'default')"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = InlineEditRAG(
        documents_path=args.documents,
        use_openai=args.openai,
        ollama_model=args.model,
        collection_name=args.collection
    )
    
    # Index documents
    rag.index_documents(
        force_reindex=args.reindex,
        incremental=not args.no_incremental
    )
    
    # Setup QA chain
    rag.setup_qa_chain()
    
    # Start chat
    rag.chat()


if __name__ == "__main__":
    main()

