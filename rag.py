import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq 

load_dotenv() 
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "qwen/qwen3.8-27b"


def chunk(text, size=40, overlap=10):
    words = text.split()
    step = size - overlap

    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):  
            break
    return chunks


text = open("company_policy_rag_test.txt", encoding="utf-8").read()
chunks = chunk(text)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
vectors = embedder.encode(chunks, normalize_embeddings=True)
print(vectors.shape)


def retrieve(question, k=3):
    q_vec = embedder.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q_vec 
    top_idx = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_idx]


def answer(question):
    context = "\n\n".join(retrieve(question))
    prompt = (
        "Answer the question using only the context below.\n"
        "If the answer isn't there, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    reply = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return reply.choices[0].message.content


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        print(answer(q))