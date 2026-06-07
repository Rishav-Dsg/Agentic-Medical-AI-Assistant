from src.llm.llm import llm
from src.llm.prompts import report_prompt

chain = report_prompt | llm

def generate_report(
    predictions,
    context
):

    response = chain.invoke(
        {
            "predictions": predictions,
            "context": context
        }
    )

    return response.content