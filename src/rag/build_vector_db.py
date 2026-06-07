from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from rag.medical_knowledge import medical_docs

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [
    doc["text"]
    for doc in medical_docs
]

embeddings = model.encode(texts)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(
    index,
    "medical_index.faiss"
)

print("Vector DB created!")