import json
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import graph_state
from .prompt import ROADMAP_PROMPT
from core.llm import get_llm


def generate_roadmap(state: graph_state) -> dict:
    """
    Roadmap generator node.

    Takes the sub_tasks from the orchestrator and generates a detailed
    7-day learning roadmap with a concept map. Updates the state with
    the generated roadmap and concept_map.

    Args:
        state: The current graph state containing sub_tasks from orchestrator.

    Returns:
        A partial state update with 'roadmap' and 'concept_map' fields populated.
    """
    llm = get_llm()

    sub_tasks = state.sub_tasks
    user_query = state.user_query

    messages = [
        SystemMessage(content=ROADMAP_PROMPT),
        HumanMessage(content=json.dumps({
            "user_query": user_query,
            "sub_tasks": sub_tasks
        }, indent=2))
    ]

    response = llm.invoke(messages)

    content = response.content
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]

    try:
        roadmap = json.loads(content.strip())
    except json.JSONDecodeError:
        roadmap = {"raw_response": content}

    concept_map = {}
    for day_key, day_data in roadmap.items():
        if isinstance(day_data, dict) and "theme" in day_data:
            concept_map[day_key] = {
                "theme": day_data.get("theme", ""),
                "objective": day_data.get("objective", "")
            }
    print("Generated Roadmap:", roadmap)
    print("Extracted Concept Map:", concept_map)
    return {
        "roadmap": roadmap,
        "concept_map": concept_map
    }