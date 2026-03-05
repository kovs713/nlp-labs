import os
import re
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from sentence_transformers import SentenceTransformer

load_dotenv()


def clean_text(text: str) -> str:
    text = text.replace("<span></span><code>", "```").replace("</code>", "```")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace('tabindex="0"', "")
    return text


class GitDocLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        with open(self.file_path, encoding="utf-8") as f:
            content = f.read()
        return [Document(page_content=content, metadata={"source": self.file_path})]


def load_documents(url: str = "", local_file: str = "git-doc.md"):
    if url:
        loader = WebBaseLoader(url)
        print(f"Loading from web: {url}")
    else:
        loader = GitDocLoader(local_file)
        print(f"Loading from file: {local_file}")
    return loader.load()


def split_documents(docs, chunk_size: int = 500, chunk_overlap: int = 100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_documents(docs)


class TransformersEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()


def create_vector_store(docs, embeddings):
    store = InMemoryVectorStore(embeddings)
    store.add_documents(docs)
    return store


class RAGAgent:
    def __init__(self, vector_store):
        self.vector_store = vector_store

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Set GROQ_API_KEY environment variable")

        self.model = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=SecretStr(api_key),
        )

        self.system_prompt = (
            "Answer questions strictly based on the provided context.\n"
            "If the context doesn't contain the answer, say so.\n"
            "Answer in the same language as the question.\n\n"
            "CONTEXT:\n{context}\n\n"
            "QUESTION: {question}\n\n"
            "ANSWER:"
        )

    def retrieve(self, query: str, k: int = 5) -> str:
        # hybrid search semantic + keyword matching
        docs = self.vector_store.similarity_search(query, k=k * 2)

        # also search for English terms (git commands)
        en_terms = re.findall(r"git\s+\w+|ssh-keygen|ssh", query, re.IGNORECASE)
        for term in en_terms:
            docs.extend(self.vector_store.similarity_search(term, k=2))

        # deduplicate and clean
        seen = set()
        cleaned: list[str] = []
        for doc in docs:
            text = clean_text(doc.page_content)
            if text not in seen:
                cleaned.append(text)
                seen.add(text)

        return "\n\n".join(cleaned[:k])

    def ask(self, question: str) -> str:
        context = self.retrieve(question)
        prompt = self.system_prompt.format(context=context, question=question)
        response = self.model.invoke(prompt)
        return str(response.content)


def main():
    if not os.getenv("GROQ_API_KEY"):
        print("Error: Set GROQ_API_KEY environment variable")
        return

    print("\n[1/4] Loading documents...")
    docs = load_documents(local_file="git-doc.md")
    print(f"Loaded {len(docs)} document(s)")

    print("\n[2/4] Splitting documents...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("\n[3/4] Creating embeddings and vector store...")
    embeddings = TransformersEmbeddings()
    vector_store = create_vector_store(chunks, embeddings)
    print("Vector store created")

    print("\n[4/4] Creating RAG agent...")
    agent = RAGAgent(vector_store)
    print("RAG agent created")

    questions = [
        "Что делает команда git init?",
        "Как создать новую ветку в git?",
        "Как сгенерировать SSH ключ?",
        "Что делает git commit -m?",
    ]

    for q in questions:
        print(f"Q: {q}")
        answer = agent.ask(q)
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
