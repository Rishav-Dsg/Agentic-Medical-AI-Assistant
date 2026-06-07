from src.rag.retriever import retrieve

results = retrieve(
    "Cardiomegaly"
)

for doc in results:
    print()
    print(doc)