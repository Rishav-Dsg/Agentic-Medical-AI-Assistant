from langchain_core.prompts import ChatPromptTemplate

report_prompt = ChatPromptTemplate.from_template(
"""
You are a radiology assistant.

Predictions:
{predictions}

Retrieved Context:
{context}

Generate:

1. Findings
2. Impression
3. Recommendation

State that this report is AI-generated.
"""
)