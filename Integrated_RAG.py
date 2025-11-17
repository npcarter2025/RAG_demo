#!/usr/bin/env python3
"""
Integrated RAG CLI - RAG system with support for all file types
Features:
- All features from Inline_Edit_Rag.py
- Support for multiple programming languages:
  - TypeScript (.ts, .tsx)
  - JavaScript (.js, .jsx)
  - C/C++ (.c, .cpp, .cc, .cxx, .h, .hpp)
  - SystemVerilog (.sv, .svh)
  - Verilog (.v, .vh)
  - VHDL (.vhd, .vhdl)
  - Python (.py)
  - Perl (.pl, .pm, .pod)
  - Tcl/Tk (.tcl, .tk)
  - Build systems:
    - Makefile (Makefile, .mk, .make)
    - CMake (CMakeLists.txt, .cmake)
  - ASIC Physical Design formats:
    - LEF (.lef) - Library Exchange Format
    - DEF (.def) - Design Exchange Format
    - SPEF (.spef) - Standard Parasitic Exchange Format
    - SDC (.sdc) - Synopsys Design Constraints
    - LIB (.lib) - Liberty timing library
    - SDF (.sdf) - Standard Delay Format
    - SPICE (.sp, .spice, .cir) - Circuit simulation
    - CDL (.cdl) - Circuit Description Language
    - UPF (.upf) - Unified Power Format
    - CPF (.cpf) - Common Power Format
    - GDSII (.gds, .gds2) - Layout database
  - Text files (.txt, .md)
  - And more!
- Language-aware chunking and metadata
- Inline editing of any supported file type
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


class IntegratedRAG:
    # Language mapping: extension -> language name
    LANGUAGE_MAP = {
        # Python
        '.py': 'python',
        # TypeScript
        '.ts': 'typescript',
        '.tsx': 'typescript',
        # JavaScript
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.mjs': 'javascript',
        # C/C++
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.hxx': 'cpp',
        # SystemVerilog/Verilog
        '.sv': 'systemverilog',
        '.svh': 'systemverilog',
        '.v': 'verilog',
        '.vh': 'verilog',
        # VHDL
        '.vhd': 'vhdl',
        '.vhdl': 'vhdl',
        # Text/Markdown
        '.txt': 'text',
        '.md': 'markdown',
        '.rst': 'text',
        # Other common languages
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'bash',
        '.fish': 'bash',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.sql': 'sql',
        '.r': 'r',
        '.m': 'matlab',
        # Perl
        '.pl': 'perl',
        '.pm': 'perl',
        '.pod': 'perl',
        # Tcl/Tk
        '.tcl': 'tcl',
        '.tk': 'tcl',
        # Build systems
        '.mk': 'makefile',  # Makefile
        '.make': 'makefile',  # Makefile
        '.cmake': 'cmake',  # CMake script
        # ASIC Physical Design formats
        '.lef': 'lef',  # Library Exchange Format
        '.def': 'def',  # Design Exchange Format
        '.spef': 'spef',  # Standard Parasitic Exchange Format
        '.sdc': 'sdc',  # Synopsys Design Constraints
        '.tlf': 'tlf',  # Timing Library Format
        '.lib': 'lib',  # Liberty timing library
        '.sdf': 'sdf',  # Standard Delay Format
        '.sp': 'spice',  # SPICE netlist
        '.spice': 'spice',  # SPICE netlist
        '.cir': 'spice',  # SPICE circuit
        '.cdl': 'cdl',  # Circuit Description Language
        '.upf': 'upf',  # Unified Power Format
        '.cpf': 'cpf',  # Common Power Format
        '.gds': 'gds',  # GDSII layout
        '.gds2': 'gds',  # GDSII layout
        '.mw': 'milkyway',  # Cadence Milkyway database
        # Other formats
        '.lua': 'lua',
        '.clj': 'clojure',
        '.hs': 'haskell',
        '.ml': 'ocaml',
        '.fs': 'fsharp',
        '.ex': 'elixir',
        '.erl': 'erlang',
    }
    
    def __init__(
        self,
        documents_path: str = "documents",
        use_openai: bool = False,
        ollama_model: str = "gemma3:1b",
        openai_model: str = "gpt-3.5-turbo",
        collection_name: str = "default",
        memory_file: str = ".rag_memory.json",
        log_file: Optional[str] = ".rag_conversation.log"
    ):
        """
        Initialize the Integrated RAG system with multi-language support.
        
        Args:
            documents_path: Path to a file or directory containing files of any supported type
            use_openai: If True, use OpenAI API. If False, use local LLM (Ollama)
            ollama_model: Ollama model name (default: "gemma3:1b")
            openai_model: OpenAI model name (default: "gpt-3.5-turbo"). Cheaper options: "gpt-4o-mini", "gpt-3.5-turbo"
            collection_name: Name of the ChromaDB collection (allows multiple indexes)
            memory_file: Path to save conversation history
            log_file: Path to log file for full conversation history (default: ".rag_conversation.log", None to disable)
        """
        self.documents_path = Path(documents_path)
        self.use_openai = use_openai
        self.ollama_model = ollama_model
        self.openai_model = openai_model
        self.collection_name = collection_name
        self.memory_file = Path(memory_file)
        self.log_file = Path(log_file) if log_file else None
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
    
    def get_language_from_extension(self, file_path: Path) -> str:
        """Get language name from file extension or special filename."""
        # Handle special filenames (no extension)
        filename_lower = file_path.name.lower()
        if filename_lower == 'makefile' or filename_lower.startswith('makefile.'):
            return 'makefile'
        if filename_lower == 'cmakelists.txt':
            return 'cmake'
        
        # Handle regular extensions
        ext = file_path.suffix.lower()
        return self.LANGUAGE_MAP.get(ext, 'text')
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions."""
        return list(self.LANGUAGE_MAP.keys())
    
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
    
    def log_conversation(self, question: str, answer: str, metadata: Optional[Dict] = None):
        """Log conversation to file with timestamp and metadata."""
        if not self.log_file:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract sources from answer if present
            sources = None
            if "📄 Sources:" in answer:
                sources_line = answer.split("📄 Sources:")[-1].strip()
                sources = sources_line.split(", ") if sources_line else None
                # Remove sources from answer for cleaner log
                answer_clean = answer.split("📄 Sources:")[0].strip()
            else:
                answer_clean = answer
            
            # Check if general knowledge was used
            used_general_knowledge = "💡" in answer
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Model: {self.openai_model if self.use_openai else self.ollama_model}\n")
                if metadata:
                    f.write(f"Filter: {metadata}\n")
                f.write("-"*80 + "\n")
                f.write(f"QUESTION:\n{question}\n\n")
                f.write(f"ANSWER:\n{answer_clean}\n")
                if sources:
                    f.write(f"\nSources: {', '.join(sources)}\n")
                if used_general_knowledge:
                    f.write("\n[Used general knowledge fallback]\n")
                f.write("="*80 + "\n")
        except Exception as e:
            print(f"⚠️  Could not write to log file: {e}")
    
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
        Note: Currently only supports Python. Can be extended for other languages.
        """
        language = self.get_language_from_extension(file_path)
        if language != 'python':
            return None  # Only Python AST parsing supported for now
        
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
    
    def parse_code_file(self, file_path: Path, content: str) -> List[Document]:
        """
        Parse a code file (non-Python) with language-aware chunking.
        Returns list of Document objects with metadata.
        """
        language = self.get_language_from_extension(file_path)
        
        # Use text-based chunking for non-Python files
        # Could be enhanced with language-specific parsers in the future
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        text_chunks = text_splitter.split_text(content)
        
        chunks = []
        for i, chunk in enumerate(text_chunks):
            chunks.append(Document(
                page_content=chunk,
                metadata={
                    "source": str(file_path),
                    "type": "code",
                    "language": language,
                    "file_name": file_path.name,
                    "chunk_index": i
                }
            ))
        
        return chunks
    
    def load_documents(self, incremental: bool = True) -> Tuple[List[Document], List[str]]:
        """
        Load files of all supported types with metadata and incremental indexing support.
        Returns tuple of (documents, file_paths).
        """
        documents = []
        file_paths = []
        supported_extensions = self.get_supported_extensions()
        new_files = []
        changed_files = []
        
        # Check if it's a single file
        if self.documents_path.is_file():
            file_path = self.documents_path
            ext = file_path.suffix.lower()
            filename_lower = file_path.name.lower()
            
            # Check if it's a supported extension or special filename
            is_supported = (ext in supported_extensions or 
                          filename_lower == 'makefile' or 
                          filename_lower.startswith('makefile.') or
                          filename_lower == 'cmakelists.txt')
            
            if is_supported:
                file_hash = self.get_file_hash(file_path)
                stored_hash = self.file_hashes.get(str(file_path))
                
                if not incremental or file_hash != stored_hash:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            if content.strip():
                                language = self.get_language_from_extension(file_path)
                                
                                if language == 'python':
                                    # Use AST parsing for Python files
                                    chunks = self.parse_python_file(file_path, content)
                                    documents.extend(chunks)
                                else:
                                    # Use language-aware chunking for other files
                                    chunks = self.parse_code_file(file_path, content)
                                    documents.extend(chunks)
                                
                                file_paths.append(str(file_path))
                                self.file_hashes[str(file_path)] = file_hash
                                
                                if stored_hash is None:
                                    new_files.append(file_path)
                                    print(f"📄 New: {file_path} ({language})")
                                else:
                                    changed_files.append(file_path)
                                    print(f"🔄 Updated: {file_path} ({language})")
                                print(f"   Loaded: {file_path}")
                            else:
                                print(f"⚠️  File '{file_path}' is empty")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                else:
                    print(f"⏭️  Skipped (unchanged): {file_path}")
            else:
                print(f"⚠️  File '{file_path}' is not a supported file type")
                print(f"   Supported extensions: {', '.join(sorted(set([ext for ext in supported_extensions[:20]])))}...")
        
        # Otherwise, treat it as a directory
        elif self.documents_path.is_dir():
            # Find all supported files
            for ext in supported_extensions:
                pattern = f"*{ext}"
                for file_path in self.documents_path.rglob(pattern):
                    # Skip if already processed
                    if str(file_path) in file_paths:
                        continue
                    file_hash = self.get_file_hash(file_path)
                    stored_hash = self.file_hashes.get(str(file_path))
                    
                    if not incremental or file_hash != stored_hash:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                if content.strip():
                                    language = self.get_language_from_extension(file_path)
                                    
                                    if language == 'python':
                                        # Use AST parsing for Python files
                                        chunks = self.parse_python_file(file_path, content)
                                        documents.extend(chunks)
                                    else:
                                        # Use language-aware chunking for other files
                                        chunks = self.parse_code_file(file_path, content)
                                        documents.extend(chunks)
                                    
                                    file_paths.append(str(file_path))
                                    self.file_hashes[str(file_path)] = file_hash
                                    
                                    if stored_hash is None:
                                        new_files.append(file_path)
                                        print(f"📄 New: {file_path} ({language})")
                                    else:
                                        changed_files.append(file_path)
                                        print(f"🔄 Updated: {file_path} ({language})")
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
                    else:
                        print(f"⏭️  Skipped (unchanged): {file_path}")
            
            # Also search for special filenames (no extension)
            special_filenames = ['Makefile', 'CMakeLists.txt']
            for special_name in special_filenames:
                for file_path in self.documents_path.rglob(special_name):
                    # Skip if already processed
                    if str(file_path) in file_paths:
                        continue
                    
                    file_hash = self.get_file_hash(file_path)
                    stored_hash = self.file_hashes.get(str(file_path))
                    
                    if not incremental or file_hash != stored_hash:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                if content.strip():
                                    language = self.get_language_from_extension(file_path)
                                    
                                    # Use language-aware chunking
                                    chunks = self.parse_code_file(file_path, content)
                                    documents.extend(chunks)
                                    
                                    file_paths.append(str(file_path))
                                    self.file_hashes[str(file_path)] = file_hash
                                    
                                    if stored_hash is None:
                                        new_files.append(file_path)
                                        print(f"📄 New: {file_path} ({language})")
                                    else:
                                        changed_files.append(file_path)
                                        print(f"🔄 Updated: {file_path} ({language})")
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
            print(f"Using OpenAI: {self.openai_model}")
            llm = ChatOpenAI(
                model=self.openai_model,
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
        self.llm = llm  # Store LLM for fallback
        print("✅ QA chain ready")
    
    def ask(self, question: str, filter_metadata: Optional[Dict] = None) -> str:
        """
        Ask a question and get an answer with retry logic and metadata filtering.
        
        Args:
            question: The question to ask
            filter_metadata: Optional metadata filter (e.g., {"file_name": "utils.py"}, {"language": "typescript"})
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
                # Notify user about API call (RAG always uses API when OpenAI is enabled)
                if self.use_openai and attempt == 0:
                    print("   🔄 Querying OpenAI API (searching documents + generating answer)...")
                
                result = self.qa_chain.invoke({"question": question})
                answer = result["answer"]
                
                # Restore original retriever if we changed it
                if filter_metadata and self.vectorstore:
                    self.qa_chain.retriever = original_retriever
                
                # Check if answer indicates no knowledge
                answer_lower = answer.lower().strip()
                dont_know_phrases = [
                    "i don't know", "i do not know", "i cannot", "i can't",
                    "i don't have", "i do not have", "unable to", "no information",
                    "not available", "not found in", "not in the"
                ]
                
                is_dont_know = any(phrase in answer_lower for phrase in dont_know_phrases)
                
                # Show sources with metadata
                sources = result.get("source_documents", [])
                has_sources = len(sources) > 0
                
                if sources:
                    source_info = []
                    for doc in sources:
                        meta = doc.metadata if hasattr(doc, "metadata") else {}
                        source_name = meta.get("file_name", Path(meta.get("source", "")).name if meta.get("source") else "unknown")
                        source_type = meta.get("type", "text")
                        language = meta.get("language", "unknown")
                        if meta.get("name"):
                            source_info.append(f"{source_name} ({language}, {source_type}: {meta['name']})")
                        else:
                            source_info.append(f"{source_name} ({language})")
                    if source_info:
                        answer += f"\n\n📄 Sources: {', '.join(set(source_info))}"
                
                # Fallback to general knowledge if RAG didn't help
                # This handles cases where:
                # 1. No sources found (general knowledge questions)
                # 2. Sources found but answer is still "I don't know" (irrelevant documents)
                if is_dont_know and hasattr(self, 'llm') and self.llm:
                    # Try direct LLM call for general knowledge questions
                    try:
                        # Notify user that we're making an API call
                        if self.use_openai:
                            print("   🔄 Making OpenAI API call (RAG didn't find relevant documents)...")
                        
                        # Check if question is about current date/time
                        question_lower = question.lower()
                        is_date_question = any(word in question_lower for word in [
                            "today", "date", "what day", "current date", "what's the date",
                            "what date", "now", "current time", "what time"
                        ])
                        
                        # Add current date/time context if needed
                        enhanced_question = question
                        if is_date_question:
                            current_datetime = datetime.now()
                            current_date_str = current_datetime.strftime("%A, %B %d, %Y")
                            # Try to get timezone, fallback to local time if not available
                            try:
                                import time
                                timezone_name = time.tzname[0] if time.tzname else "local time"
                            except:
                                timezone_name = "local time"
                            current_time_str = current_datetime.strftime(f"%I:%M %p ({timezone_name})")
                            enhanced_question = f"""Current date and time information:
- Date: {current_date_str}
- Time: {current_time_str}
- Day of week: {current_datetime.strftime('%A')}

User question: {question}

Please answer the user's question using the current date/time information provided above."""
                        
                        direct_response = self.llm.invoke(enhanced_question)
                        if direct_response and hasattr(direct_response, 'content'):
                            fallback_answer = direct_response.content
                            # Only use fallback if it's different and more helpful
                            fallback_lower = fallback_answer.lower()
                            if not any(phrase in fallback_lower for phrase in dont_know_phrases) and len(fallback_answer) > 20:
                                source_note = "no relevant documents found" if not has_sources else "retrieved documents weren't relevant"
                                return f"{fallback_answer}\n\n💡 (Answered using general knowledge - {source_note})"
                    except Exception:
                        pass  # If fallback fails, return original answer
                
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
    
    def edit_file_inline(self, file_path: Path, instruction: str) -> Tuple[bool, str]:
        """
        Edit a file based on a natural language instruction.
        Uses the LLM to generate the edit, then applies it with diff preview.
        Supports all file types.
        """
        if not self.qa_chain:
            return False, "❌ QA chain not set up. Please index documents first."
        
        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            return False, f"❌ Error reading file: {e}"
        
        # Detect language
        language = self.get_language_from_extension(file_path)
        
        # Ask LLM to generate the edited version
        prompt = f"""You are a code editor. Edit the following {language} file according to the instruction.

CURRENT FILE ({file_path.name}):
```{language}
{file_content}
```

INSTRUCTION: {instruction}

REQUIREMENTS:
1. You MUST modify the code according to the instruction
2. Return the COMPLETE file with your changes
3. Preserve all existing code that doesn't need to change
4. Only modify what the instruction asks for
5. Keep the same file structure, imports, and formatting
6. Maintain the correct syntax for {language}

Return ONLY the complete edited code in a markdown code block:
```{language}
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
            
            # Get the code (should be the full file)
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
    
    def chat(self):
        """Start an interactive chat session with inline editing capabilities."""
        if not self.qa_chain:
            print("❌ QA chain not set up. Please index documents first.")
            return
        
        # Initialize log file with session header
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write("\n" + "="*80 + "\n")
                    f.write(f"NEW SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Model: {self.openai_model if self.use_openai else self.ollama_model}\n")
                    f.write(f"Collection: {self.collection_name}\n")
                    f.write("="*80 + "\n")
                print(f"📝 Conversation logging to: {self.log_file}")
            except Exception as e:
                print(f"⚠️  Could not initialize log file: {e}")
        
        print("\n" + "="*60)
        print("💬 Integrated RAG - Multi-Language Chat with Code Editing")
        print("   Type 'quit' or 'exit' to end the conversation")
        print("   Type 'clear' to clear conversation history")
        print("   Type 'filter: filename.ext' to filter by file")
        print("   Type 'filter: language typescript' to filter by language")
        print("   Type 'edit: filename.ext instruction' to edit a file")
        print("   When code is generated, you'll be asked to save or display it")
        if self.log_file:
            print(f"   📝 Full conversation logged to: {self.log_file}")
        print("="*60 + "\n")
        
        current_filter = None
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ["quit", "exit", "q"]:
                    self.save_memory()
                    if self.log_file:
                        try:
                            with open(self.log_file, 'a', encoding='utf-8') as f:
                                f.write(f"\nSESSION ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                f.write("="*80 + "\n\n")
                        except:
                            pass
                    print("👋 Goodbye! Conversation history saved.")
                    if self.log_file:
                        print(f"📝 Full conversation logged to: {self.log_file}")
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
                        print("❌ Usage: edit: filename.ext instruction")
                        print("   Example: edit: app.ts Add error handling")
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
                        # Search in documents directory for all supported extensions
                        for ext in self.get_supported_extensions():
                            pattern = f"*{ext}"
                            for f in self.documents_path.rglob(pattern):
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
                    if filter_cmd.startswith("language "):
                        lang_name = filter_cmd[9:].strip()
                        current_filter = {"language": lang_name}
                        print(f"✅ Filter set to language: {lang_name}\n")
                    elif any(filter_cmd.endswith(ext) for ext in self.get_supported_extensions()):
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
                        print("❌ Invalid filter. Use 'filter: filename.ext', 'filter: language langname', or 'filter: function name'\n")
                    continue
                
                print("\n🤖 Assistant: ", end="", flush=True)
                answer = self.ask(question, filter_metadata=current_filter)
                print(answer)
                
                # Log conversation to file
                filter_meta = current_filter if current_filter else None
                self.log_conversation(question, answer, metadata=filter_meta)
                
                # Save memory after each exchange
                self.save_memory()
                
                # Check if answer contains code blocks
                code_blocks = self.extract_code_blocks(answer)
                if code_blocks:
                    code = None
                    code_lang = None
                    for lang, code_content in code_blocks:
                        if lang:
                            code = code_content
                            code_lang = lang
                            break
                    
                    if not code and code_blocks:
                        code_lang, code = code_blocks[0]
                        if not code_lang:
                            code_lang = 'text'
                    
                    if code:
                        print(f"\n💡 Found {code_lang} code block. What would you like to do?")
                        print("   [s] Save to file")
                        print("   [d] Display code only (don't save)")
                        print("   [n] Nothing (skip)")
                        
                        while True:
                            choice = input("\nYour choice (s/d/n): ").strip().lower()
                            
                            if choice == 's' or choice == 'save':
                                # Determine file extension from language
                                lang_to_ext = {
                                    'python': '.py',
                                    'typescript': '.ts',
                                    'javascript': '.js',
                                    'cpp': '.cpp',
                                    'c': '.c',
                                    'java': '.java',
                                    'go': '.go',
                                    'rust': '.rs',
                                    'ruby': '.rb',
                                    'php': '.php',
                                    'swift': '.swift',
                                    'kotlin': '.kt',
                                    'scala': '.scala',
                                    'bash': '.sh',
                                    'yaml': '.yaml',
                                    'json': '.json',
                                    'html': '.html',
                                    'css': '.css',
                                    'sql': '.sql',
                                    'systemverilog': '.sv',
                                    'verilog': '.v',
                                    'vhdl': '.vhd',
                                    'perl': '.pl',
                                    'tcl': '.tcl',
                                    # Build systems
                                    'makefile': 'Makefile',
                                    'cmake': 'CMakeLists.txt',
                                    # ASIC Physical Design
                                    'lef': '.lef',
                                    'def': '.def',
                                    'spef': '.spef',
                                    'sdc': '.sdc',
                                    'lib': '.lib',
                                    'sdf': '.sdf',
                                    'spice': '.sp',
                                    'cdl': '.cdl',
                                    'upf': '.upf',
                                    'cpf': '.cpf',
                                }
                                ext = lang_to_ext.get(code_lang.lower(), '.txt')
                                
                                filename = re.sub(r'[^\w\s-]', '', question.lower())
                                filename = re.sub(r'[-\s]+', '_', filename)
                                filename = filename[:30]
                                if not filename:
                                    filename = "generated_code"
                                filename += ext
                                
                                try:
                                    output_path = Path(".")
                                    file_path = output_path / filename
                                    
                                    if file_path.exists():
                                        print(f"⚠️  File '{file_path}' already exists. Not overwriting.\n")
                                    else:
                                        with open(file_path, 'w', encoding='utf-8') as f:
                                            f.write(code)
                                        print(f"✅ Created file: {file_path}\n")
                                except Exception as e:
                                    print(f"❌ Error creating file: {e}\n")
                                break
                            
                            elif choice == 'd' or choice == 'display':
                                print(f"\n📝 Code:\n```{code_lang}")
                                print(code)
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
                if self.log_file:
                    try:
                        with open(self.log_file, 'a', encoding='utf-8') as f:
                            f.write(f"\nSESSION ENDED (KeyboardInterrupt): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write("="*80 + "\n\n")
                    except:
                        pass
                print("\n\n👋 Goodbye! Conversation history saved.")
                if self.log_file:
                    print(f"📝 Full conversation logged to: {self.log_file}")
                break
            except EOFError:
                self.save_memory()
                if self.log_file:
                    try:
                        with open(self.log_file, 'a', encoding='utf-8') as f:
                            f.write(f"\nSESSION ENDED (EOF): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write("="*80 + "\n\n")
                    except:
                        pass
                print("\n\n👋 Goodbye! Conversation history saved.")
                if self.log_file:
                    print(f"📝 Full conversation logged to: {self.log_file}")
                break


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Integrated RAG CLI - RAG with support for all file types"
    )
    parser.add_argument(
        "--documents",
        type=str,
        default="documents",
        help="Path to a file or directory containing files of any supported type (default: 'documents')"
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
        "--openai-model",
        type=str,
        dest="openai_model",
        default="gpt-3.5-turbo",
        help="OpenAI model name (default: 'gpt-3.5-turbo'). Cheaper options: 'gpt-4o-mini' (cheapest), 'gpt-3.5-turbo'"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="default",
        help="ChromaDB collection name (allows multiple indexes, default: 'default')"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        dest="log_file",
        default=".rag_conversation.log",
        help="Path to conversation log file (default: '.rag_conversation.log', use 'none' to disable)"
    )
    
    args = parser.parse_args()
    
    # Handle log file option
    log_file = None if args.log_file.lower() == 'none' else args.log_file
    
    # Initialize RAG system
    rag = IntegratedRAG(
        documents_path=args.documents,
        use_openai=args.openai,
        ollama_model=args.model,
        openai_model=args.openai_model,
        collection_name=args.collection,
        log_file=log_file
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

