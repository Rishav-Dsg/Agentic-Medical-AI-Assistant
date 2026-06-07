from src.rag.retriever import retrieve

def retrieval_agent(state):

    predictions = state["predictions"]

    context = []

    for disease, prob in predictions.items():

        if prob > 0.5:

            docs = retrieve(
                disease,
                k=1
            )

            context.extend(docs)

    state["context"] = "\n".join(
        context
    )

    return state