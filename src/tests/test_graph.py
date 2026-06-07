from src.graph.state import MedicalState
from src.graph.medical_graph import medical_graph

state: MedicalState = {
    "image_path": "data/images/00000001_000.png",
    "predictions": {},
    "context": "",
    "report": ""
}

result = medical_graph.invoke(state)

print(result["report"])