from src.llm.report_generator import generate_report

def report_agent(state):

    state["report"] = generate_report(
        state["predictions"],
        state["context"]
    )

    return state