#!/usr/bin/env python3
"""Test script to verify all imports from main.py work correctly."""

import sys

print("Testing imports...\n")

tests = [
    ("os", lambda: __import__("os")),
    ("re", lambda: __import__("re")),
    ("collections.Counter", lambda: __import__("collections").Counter),
    ("dotenv.load_dotenv", lambda: __import__("dotenv").load_dotenv),
    (
        "langchain_core.embeddings.Embeddings",
        lambda: __import__(
            "langchain_core.embeddings", fromlist=["Embeddings"]
        ).Embeddings,
    ),
    (
        "langchain_community.document_loaders.TextLoader",
        lambda: __import__(
            "langchain_community.document_loaders", fromlist=["TextLoader"]
        ).TextLoader,
    ),
    (
        "langchain_text_splitters.RecursiveCharacterTextSplitter",
        lambda: __import__(
            "langchain_text_splitters", fromlist=["RecursiveCharacterTextSplitter"]
        ).RecursiveCharacterTextSplitter,
    ),
    (
        "langchain_community.vectorstores.FAISS",
        lambda: __import__(
            "langchain_community.vectorstores", fromlist=["FAISS"]
        ).FAISS,
    ),
    (
        "langchain_openai.ChatOpenAI",
        lambda: __import__("langchain_openai", fromlist=["ChatOpenAI"]).ChatOpenAI,
    ),
]

failed = []
for module_name, import_func in tests:
    try:
        import_func()
        print(f"OK {module_name}")
    except Exception as e:
        print(f"FAIL {module_name}")
        print(f"   Error: {type(e).__name__}: {e}\n")
        failed.append((module_name, e))

print("\n" + "=" * 50)
if failed:
    print(f"FAIL {len(failed)} import(s) failed")
    sys.exit(1)

print("OK All imports successful!")
sys.exit(0)
