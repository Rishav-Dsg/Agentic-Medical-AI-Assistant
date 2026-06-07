from src.agents.prediction_agent import prediction_agent
from src.agents.retrieval_agent import retrieval_agent
from src.agents.report_agent import report_agent

def main():

    state = {
        "image_path": "data/images/00000001_000.png"
    }

    print("\nRunning Prediction Agent...\n")

    state = prediction_agent(state)

    print("Predictions:")
    print(state["predictions"])

    print("\nRunning Retrieval Agent...\n")

    state = retrieval_agent(state)

    print("Retrieved Context:")
    print(state["context"])

    print("\nRunning Report Agent...\n")

    state = report_agent(state)

    print("\nGenerated Report:\n")
    print("=" * 80)
    print(state["report"])
    print("=" * 80)


if __name__ == "__main__":
    main()