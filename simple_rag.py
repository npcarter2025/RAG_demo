#!/usr/bin/env python3
"""
Simple RAG CLI - Chat with your .txt and .md documents
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleRAG:
    def __init__(self, documents_path: str = "documents", use_openai: bool = False, ollama_model: str = "gemma3:1b"):
        """
        Initialize the RAG system.
        
        Args:
            documents_path: Path to a file or directory containing .txt, .md, or .py files
            use_openai: If True, use OpenAI API. If False, use local LLM (Ollama)
            ollama_model: Ollama model name (default: "llama3.1:latest")
        """
        self.documents_path = Path(documents_path)
        self.use_openai = use_openai
        self.ollama_model = ollama_model
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create documents directory if it doesn't exist (only if it's a directory)
        if self.documents_path.is_dir() or not self.documents_path.exists():
            self.documents_path.mkdir(exist_ok=True)
        
        # Initialize embeddings (free, local)
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
    def load_documents(self) -> List[str]:
        """Load .txt, .md, and .py files from the documents path (file or directory)."""
        documents = []
        file_paths = []
        supported_extensions = ['.txt', '.md', '.py']
        
        # Check if it's a single file
        if self.documents_path.is_file():
            file_path = self.documents_path
            if file_path.suffix.lower() in supported_extensions:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip():
                            documents.append(content)
                            file_paths.append(str(file_path))
                            print(f"Loaded: {file_path}")
                        else:
                            print(f"⚠️  File '{file_path}' is empty")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            else:
                print(f"⚠️  File '{file_path}' is not a supported file type (.txt, .md, or .py)")
        # Otherwise, treat it as a directory
        elif self.documents_path.is_dir():
            # Find all .txt, .md, and .py files
            for ext in ["*.txt", "*.md", "*.py"]:
                for file_path in self.documents_path.rglob(ext):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            if content.strip():  # Only add non-empty files
                                documents.append(content)
                                file_paths.append(str(file_path))
                                print(f"Loaded: {file_path}")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        else:
            print(f"⚠️  Path '{self.documents_path}' does not exist")
            return []
        
        if not documents:
            print(f"\n⚠️  No supported files (.txt, .md, or .py) found at '{self.documents_path}'")
            if self.documents_path.is_dir():
                print(f"   Please add some documents to that directory and try again.")
            else:
                print(f"   Please check that the file path is correct.")
            return []
        
        print(f"\n✅ Loaded {len(documents)} document(s)")
        return documents
    
    def create_chunks(self, documents: List[str]) -> List:
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.create_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def index_documents(self, force_reindex: bool = False):
        """Index documents into the vector store."""
        persist_directory = "./chroma_db"
        
        # Check if vectorstore already exists
        if not force_reindex and Path(persist_directory).exists():
            print(f"Loading existing vector store from {persist_directory}...")
            try:
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                print("✅ Loaded existing index")
                return
            except Exception as e:
                print(f"Error loading existing index: {e}")
                print("Creating new index...")
        
        # Load and process documents
        documents = self.load_documents()
        if not documents:
            return
        
        # Create chunks
        chunks = self.create_chunks(documents)
        
        # Create vector store
        print("Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        print(f"✅ Indexed documents in {persist_directory}")
    
    def setup_qa_chain(self):
        """Set up the question-answering chain."""
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
                print(f"   You can check with: ollama list")
                return
        
        # Create QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )
        print("✅ QA chain ready")
    
    def ask(self, question: str) -> str:
        """Ask a question and get an answer."""
        if not self.qa_chain:
            return "❌ QA chain not set up. Please index documents first."
        
        try:
            result = self.qa_chain.invoke({"question": question})
            answer = result["answer"]
            
            # Optionally show sources
            sources = result.get("source_documents", [])
            if sources:
                source_files = set()
                for doc in sources:
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        source_files.add(Path(doc.metadata["source"]).name)
                if source_files:
                    answer += f"\n\n📄 Sources: {', '.join(source_files)}"
            
            return answer
        except Exception as e:
            error_msg = str(e)
            if "500" in error_msg or "model runner" in error_msg.lower():
                return f"❌ Ollama error: The model may not be loaded or there's a resource issue.\n   Try: ollama run {self.ollama_model}\n   Or check Ollama server logs.\n   Error: {error_msg}"
            return f"❌ Error: {error_msg}"
    
    def extract_code_blocks(self, text: str) -> List[tuple]:
        """
        Extract code blocks from markdown-formatted text.
        Returns list of (language, code) tuples.
        """
        # Pattern to match code blocks: ```language\ncode\n```
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [(lang.strip() if lang else '', code.strip()) for lang, code in matches]
    
    def create_python_file(self, filename: str, code: str, output_dir: str = ".") -> str:
        """
        Create a Python file with the given code.
        
        Args:
            filename: Name of the file (with or without .py extension)
            code: Python code to write
            output_dir: Directory to save the file (default: current directory)
        
        Returns:
            Path to created file or error message
        """
        try:
            # Ensure .py extension
            if not filename.endswith('.py'):
                filename += '.py'
            
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Full file path
            file_path = output_path / filename
            
            # Check if file exists
            if file_path.exists():
                return f"⚠️  File '{file_path}' already exists. Not overwriting."
            
            # Write code to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            return f"✅ Created Python file: {file_path}"
        except Exception as e:
            return f"❌ Error creating file: {e}"
    
    def chat(self):
        """Start an interactive chat session."""
        if not self.qa_chain:
            print("❌ QA chain not set up. Please index documents first.")
            return
        
        print("\n" + "="*60)
        print("💬 Chat with your documents!")
        print("   Type 'quit' or 'exit' to end the conversation")
        print("   Type 'clear' to clear conversation history")
        print("   When code is generated, you'll be asked to save or display it")
        print("="*60 + "\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                
                if question.lower() == "clear":
                    self.memory.clear()
                    print("✅ Conversation history cleared\n")
                    continue
                
                print("\n🤖 Assistant: ", end="", flush=True)
                answer = self.ask(question)
                print(answer)
                
                # Check if answer contains code blocks and ask what to do
                code_blocks = self.extract_code_blocks(answer)
                if code_blocks:
                    # Find Python code block
                    python_code = None
                    code_lang = None
                    for lang, code in code_blocks:
                        if lang.lower() in ['python', 'py', ''] or not lang:
                            python_code = code
                            code_lang = lang if lang else 'python'
                            break
                    
                    if not python_code and code_blocks:
                        # Use first code block if no Python found
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
                                # Generate filename from question
                                filename = re.sub(r'[^\w\s-]', '', question.lower())
                                filename = re.sub(r'[-\s]+', '_', filename)
                                filename = filename[:30]  # Limit length
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
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simple RAG CLI - Chat with your .txt, .md, and .py documents"
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
        help="Force reindexing of documents"
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
        help="Ollama model name (default: 'gemma3:1b'). Examples: llama3.1:latest, mistral, qwen2.5"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = SimpleRAG(
        documents_path=args.documents,
        use_openai=args.openai,
        ollama_model=args.model
    )
    
    # Index documents
    rag.index_documents(force_reindex=args.reindex)
    
    # Setup QA chain
    rag.setup_qa_chain()
    
    # Start chat
    rag.chat()


if __name__ == "__main__":
    main()

