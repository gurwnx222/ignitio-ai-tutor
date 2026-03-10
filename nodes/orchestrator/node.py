import json
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import graph_state
from .prompt import SYSTEM_PROMPT
from core.llm import get_llm


def orchestrator_node(state: graph_state) -> dict:
    """
    Orchestrator node.

    Takes the user_query and generates a structured 7-day sub_tasks plan.
    The sub_tasks contain instructions for the roadmap worker and other
    downstream nodes.

    Args:
        state: The current graph state containing user_query.

    Returns:
        A partial state update with 'sub_tasks' field populated.
    """
    llm = get_llm()

    user_query = state.user_query

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query)
    ]

    response = llm.invoke(messages)

    # Parse JSON response
    content = response.content
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]

    try:
        sub_tasks = json.loads(content.strip())
    except json.JSONDecodeError:
        # Fallback: create minimal structure
        sub_tasks = {
            f"day_{i}": {
                "title": f"Day {i} Learning",
                "theme": "LangChain Fundamentals",
                "roadmap_instruction": "Create a focused learning plan",
                "concept_instruction": "Map key concepts",
                "explain_instruction": "Explain clearly",
                "code_instruction": "Provide working code examples",
                "tasks": []
            }
            for i in range(1, 8)
        }

    print("Orchestrator generated sub_tasks for", len(sub_tasks), "days")

    return {
        "sub_tasks": sub_tasks
    }
