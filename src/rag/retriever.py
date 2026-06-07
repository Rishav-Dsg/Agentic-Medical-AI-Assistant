from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from src.rag.medical_knowledge import medical_docs

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

index = faiss.read_index(
    "./models/medical_index.faiss"
)

def retrieve(query, k=2):

    embedding = model.encode(
        [query]
    )

    embedding = np.array(
        embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        embedding,
        k
    )

    results = []

    for idx in indices[0]:
        results.append(
            medical_docs[idx]["text"]
        )

    return results