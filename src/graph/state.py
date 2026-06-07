from typing import TypedDict

class MedicalState(TypedDict, total=False):
    image_path: str
    predictions: dict
    context: str
    report: str