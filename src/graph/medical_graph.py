from langgraph.graph import StateGraph
from langgraph.graph import END

from src.graph.state import MedicalState

from src.agents.prediction_agent import prediction_agent
from src.agents.retrieval_agent import retrieval_agent
from src.agents.report_agent import report_agent

builder = StateGraph(MedicalState)

builder.add_node(
    "prediction_agent",
    prediction_agent
)

builder.add_node(
    "retrieval_agent",
    retrieval_agent
)

builder.add_node(
    "report_agent",
    report_agent
)

builder.set_entry_point(
    "prediction_agent"
)

builder.add_edge(
    "prediction_agent",
    "retrieval_agent"
)

builder.add_edge(
    "retrieval_agent",
    "report_agent"
)

builder.add_edge(
    "report_agent",
    END
)

medical_graph = builder.compile()