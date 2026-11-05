"""ShopKart E-Commerce RAG Assistant (NumPy cosine similarity + Groq LLM).

Run:   python rag_shopkart.py          (interactive chat)
       python rag_shopkart.py --test   (runs all 23 test questions)
"""
# ===== SETUP =====
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()  # reads GROQ_API_KEY from the .env file

KB_DIR = Path("kb")
KB_FILES = [
    "products.txt",
    "delivery_policy.txt",
    "return_refund_policy.txt",
    "payment_policy.txt",
]
CHUNK_SIZE = 50    # words per chunk
OVERLAP = 10       # words shared between neighbouring chunks
TOP_K = 4          # chunks sent to the LLM
MIN_SCORE = 0.25   # similarity threshold (below this = "not found")
LLM_MODEL = "llama-3.3-70b-versatile"
NOT_FOUND = ("I'm sorry, I couldn't find that information "
             "in the ShopKart knowledge base.")

_client = None


def get_client():
    """Create the Groq client the first time it is needed."""
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


# ===== TASK 1: load_documents =====
def load_documents(folder=KB_DIR):
    """Read the four .txt files. Returns [{'source': filename, 'text': str}]."""
    documents = []
    for name in KB_FILES:
        path = Path(folder) / name
        text = path.read_text(encoding="utf-8")
        documents.append({"source": name, "text": text})
    return documents


# ===== TASK 2: chunk =====
def chunk(text, size=50, overlap=10):
    """Split text into word-based chunks of `size` words, sharing `overlap`."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):   # this window already reached the end
            break
    return chunks


def build_records(documents):
    """Chunk every document and keep the filename as metadata."""
    records = []
    for doc in documents:
        for i, piece in enumerate(chunk(doc["text"], CHUNK_SIZE, OVERLAP)):
            records.append({"source": doc["source"], "chunk_id": i, "text": piece})
    return records


# ===== TASK 3: embeddings =====
embedder = SentenceTransformer("all-MiniLM-L6-v2")

records = build_records(load_documents())
vectors = embedder.encode(
    [r["text"] for r in records],
    normalize_embeddings=True,    # unit-length vectors
)
print(f"Loaded {len(records)} chunks, vectors shape = {vectors.shape}")


# ===== TASK 4: retrieve =====
def retrieve(question, top_k=TOP_K, min_score=MIN_SCORE):
    """Return the top_k most similar chunks (with score) above min_score."""
    q_vec = embedder.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q_vec                    # cosine similarity per chunk
    best = np.argsort(scores)[::-1][:top_k]     # indices, highest score first
    return [
        {**records[i], "score": float(scores[i])}
        for i in best
        if scores[i] >= min_score
    ]


# ===== TASK 5 + 6: answer =====
SYSTEM_PROMPT = (
    "You are the ShopKart shopping assistant. "
    "Answer ONLY using the context provided by the user message. "
    "Never use outside knowledge and never guess prices, specs or policies. "
    "If the context does not contain the answer, reply exactly with: "
    + NOT_FOUND +
    " Keep answers short and clear. For comparisons, mention both products."
)


def answer(question):
    hits = retrieve(question)

    # Task 6, layer 1: nothing similar enough was retrieved
    if not hits:
        return {"answer": NOT_FOUND, "sources": []}

    context = "\n\n".join(f"[Source: {h['source']}]\n{h['text']}" for h in hits)
    reply = get_client().chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    text = reply.choices[0].message.content.strip()

    # Task 6, layer 2: the LLM itself said the context has no answer
    if text.startswith("I'm sorry, I couldn't find"):
        return {"answer": text, "sources": []}

    sources = sorted({h["source"] for h in hits})
    return {"answer": text, "sources": sources}


def ask(question):
    result = answer(question)
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    if result["sources"]:
        print("Sources: " + ", ".join(result["sources"]))


# ===== RUN =====
TEST_QUESTIONS = [
    "What is the price of Samsung Galaxy M35?",
    "Which smartphones have 8GB RAM?",
    "Compare Samsung Galaxy M35 and Redmi Note 14.",
    "Which products are currently out of stock?",
    "Which laptop has 16GB RAM?",
    "What is the battery capacity of Samsung Galaxy M35?",
    "How many days does standard delivery take?",
    "Is express delivery available?",
    "How can I track my order?",
    "What happens if my delivery is delayed?",
    "What is the return window?",
    "How long does a refund take?",
    "Can I return a damaged product?",
    "Which products cannot be returned?",
    "Does ShopKart support UPI?",
    "Is cash on delivery available?",
    "What happens if a payment fails?",
    "Which smartphone is cheaper, Samsung or Redmi?",
    "Find an in-stock product under Rs. 20,000.",
    "Recommend a smartphone with 8GB RAM and explain why.",
    "What is the price of iPhone 17 Pro?",
    "Does ShopKart sell refrigerators?",
    "What is the CEO's phone number?",
]

if __name__ == "__main__":
    if "--test" in sys.argv:
        for question in TEST_QUESTIONS:
            ask(question)
    else:
        print("ShopKart assistant ready. Type 'quit' to exit.")
        while True:
            q = input("\nYour question: ").strip()
            if q.lower() in {"", "quit", "exit"}:
                break
            ask(q)
