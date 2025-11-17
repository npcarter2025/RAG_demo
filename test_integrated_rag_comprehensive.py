#!/usr/bin/env python3
"""Comprehensive test script for Integrated_RAG.py"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from Integrated_RAG import IntegratedRAG

def create_test_files(test_dir: Path):
    """Create test files in various languages."""
    test_dir.mkdir(exist_ok=True)
    
    # Python
    (test_dir / "test.py").write_text("""
def hello_world():
    print("Hello, World!")
    return True
""")
    
    # TypeScript
    (test_dir / "test.ts").write_text("""
interface User {
    name: string;
    age: number;
}

function greet(user: User): string {
    return `Hello, ${user.name}!`;
}
""")
    
    # JavaScript
    (test_dir / "test.js").write_text("""
function calculateSum(a, b) {
    return a + b;
}

module.exports = { calculateSum };
""")
    
    # Makefile
    (test_dir / "Makefile").write_text("""
CC=gcc
CFLAGS=-Wall -g

all: test
test: test.c
\t$(CC) $(CFLAGS) -o test test.c

clean:
\trm -f test
""")
    
    # CMakeLists.txt
    (test_dir / "CMakeLists.txt").write_text("""
cmake_minimum_required(VERSION 3.10)
project(TestProject)

set(CMAKE_CXX_STANDARD 17)

add_executable(test_app main.cpp)
target_link_libraries(test_app PRIVATE some_lib)
""")
    
    # CMake script
    (test_dir / "FindPackage.cmake").write_text("""
find_path(PACKAGE_INCLUDE_DIR package.h)
find_library(PACKAGE_LIBRARY package)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Package DEFAULT_MSG
    PACKAGE_INCLUDE_DIR PACKAGE_LIBRARY
)
""")
    
    # C file
    (test_dir / "test.c").write_text("""
#include <stdio.h>

int main() {
    printf("Hello, World!\\n");
    return 0;
}
""")
    
    # Text file
    (test_dir / "readme.txt").write_text("""
This is a test project.
It contains multiple file types for testing.
""")
    
    print(f"✅ Created test files in {test_dir}")

def test_language_detection():
    """Test language detection for various file types."""
    print("\n" + "="*60)
    print("1️⃣ Testing Language Detection")
    print("="*60)
    
    rag = IntegratedRAG()
    
    test_cases = [
        ("test.py", "python"),
        ("test.ts", "typescript"),
        ("test.js", "javascript"),
        ("Makefile", "makefile"),
        ("CMakeLists.txt", "cmake"),
        ("config.cmake", "cmake"),
        ("test.c", "c"),
        ("test.cpp", "cpp"),
        ("design.sv", "systemverilog"),
        ("design.v", "verilog"),
        ("timing.sdc", "sdc"),
        ("library.lef", "lef"),
    ]
    
    all_passed = True
    for filename, expected_lang in test_cases:
        test_path = Path(filename)
        detected_lang = rag.get_language_from_extension(test_path)
        status = "✅" if detected_lang == expected_lang else "❌"
        if detected_lang != expected_lang:
            all_passed = False
        print(f"   {status} {filename:20s} → {detected_lang:15s} (expected: {expected_lang})")
    
    return all_passed

def test_indexing(test_dir: Path):
    """Test document indexing."""
    print("\n" + "="*60)
    print("2️⃣ Testing Document Indexing")
    print("="*60)
    
    try:
        rag = IntegratedRAG(
            documents_path=str(test_dir),
            use_openai=False,  # Use Ollama for testing (or set True for OpenAI)
            collection_name="test_comprehensive"
        )
        
        print("🔄 Indexing documents...")
        rag.index_documents(force_reindex=True, incremental=False)
        
        if not rag.vectorstore:
            print("❌ Vector store not created")
            return False
        
        collection = rag.vectorstore._collection
        count = collection.count()
        print(f"✅ Indexed {count} document chunks")
        
        # Check languages found
        if count > 0:
            results = collection.peek(limit=min(20, count))
            languages_found = set()
            files_found = set()
            for meta in results.get('metadatas', []):
                if meta and 'language' in meta:
                    languages_found.add(meta['language'])
                if meta and 'file_name' in meta:
                    files_found.add(meta['file_name'])
            
            print(f"🌐 Languages found: {', '.join(sorted(languages_found))}")
            print(f"📄 Files found: {', '.join(sorted(files_found))}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qa_chain(test_dir: Path):
    """Test QA chain setup."""
    print("\n" + "="*60)
    print("3️⃣ Testing QA Chain Setup")
    print("="*60)
    
    try:
        rag = IntegratedRAG(
            documents_path=str(test_dir),
            use_openai=False,  # Use Ollama for testing
            collection_name="test_comprehensive"
        )
        
        rag.index_documents(force_reindex=False, incremental=False)
        rag.setup_qa_chain()
        
        if rag.qa_chain:
            print("✅ QA chain setup successful")
            return True
        else:
            print("⚠️  QA chain not set up (may need OpenAI API key or Ollama)")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_querying(test_dir: Path, use_openai: bool = False):
    """Test querying functionality."""
    print("\n" + "="*60)
    print("4️⃣ Testing Query Functionality")
    print("="*60)
    
    if not use_openai:
        print("⚠️  Skipping query test (set use_openai=True to test)")
        return True
    
    try:
        rag = IntegratedRAG(
            documents_path=str(test_dir),
            use_openai=True,
            openai_model="gpt-4o-mini",
            collection_name="test_query",
            log_file=".rag_conversation.log"  # Enable logging
        )
        
        rag.index_documents(force_reindex=True, incremental=False)
        rag.setup_qa_chain()
        
        if not rag.qa_chain:
            print("❌ QA chain not set up")
            return False
        
        test_questions = [
            "What does the Python function do?",
            "What is in the Makefile?",
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            try:
                answer = rag.ask(question)
                print(f"✅ Got answer (length: {len(answer)} chars)")
                if "📄 Sources:" in answer:
                    print("   ✅ Sources found in answer")
                
                # Log the conversation
                rag.log_conversation(question, answer, metadata=None)
                print("   📝 Logged to conversation file")
            except Exception as e:
                print(f"❌ Error: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_filtering(test_dir: Path):
    """Test metadata filtering."""
    print("\n" + "="*60)
    print("5️⃣ Testing Language Filtering")
    print("="*60)
    
    try:
        rag = IntegratedRAG(
            documents_path=str(test_dir),
            use_openai=False,
            collection_name="test_filter"
        )
        
        rag.index_documents(force_reindex=True, incremental=False)
        
        if not rag.vectorstore:
            print("❌ Vector store not created")
            return False
        
        # Test language filter
        filter_metadata = {"language": "python"}
        retriever = rag.vectorstore.as_retriever(
            search_kwargs={"k": 5, "filter": filter_metadata}
        )
        docs = retriever.invoke("function")
        
        python_docs = [d for d in docs if d.metadata.get('language') == 'python']
        print(f"✅ Filtered to {len(python_docs)} Python documents")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("🧪 Comprehensive Test Suite for Integrated_RAG.py")
    print("="*60)
    
    # Create temporary test directory
    test_dir = Path("test_integrated_rag_files")
    
    # Log file for test session
    test_log_file = ".rag_conversation.log"
    print(f"📝 Test conversations will be logged to: {test_log_file}")
    
    try:
        # Create test files
        create_test_files(test_dir)
        
        # Run tests
        results = {
            "Language Detection": test_language_detection(),
            "Document Indexing": test_indexing(test_dir),
            "QA Chain Setup": test_qa_chain(test_dir),
            "Language Filtering": test_filtering(test_dir),
        }
        
        # Optional: Test querying (requires OpenAI API key)
        use_openai = os.getenv("OPENAI_API_KEY") is not None
        if use_openai:
            results["Query Functionality"] = test_querying(test_dir, use_openai=True)
        else:
            print("\n⚠️  Skipping query test (no OPENAI_API_KEY found)")
            print("   Set OPENAI_API_KEY in .env to test querying")
        
        # Summary
        print("\n" + "="*60)
        print("📊 Test Summary")
        print("="*60)
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {test_name:25s}: {status}")
        
        all_passed = all(results.values())
        if all_passed:
            print("\n✅ All tests passed!")
        else:
            print("\n⚠️  Some tests failed. See details above.")
        
        return all_passed
        
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up test files...")
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("✅ Cleanup complete")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

