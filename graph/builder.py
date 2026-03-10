from langgraph.graph import StateGraph, END
from graph.state import graph_state
from nodes.orchestrator.node import orchestrator_node
from nodes.roadmap.node import generate_roadmap


def build_graph():
    """
    Build and compile the LangGraph for the AI tutor.

    Flow:
        1. orchestrator_node - Generates initial sub_tasks from user_query
        2. roadmap_node - Takes sub_tasks and generates detailed 7-day roadmap

    Returns:
        Compiled LangGraph instance ready for invocation.
    """
    workflow = StateGraph(graph_state)

    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("roadmap", generate_roadmap)

    # Set entry point
    workflow.set_entry_point("orchestrator")

    # Connect orchestrator -> roadmap
    workflow.add_edge("orchestrator", "roadmap")

    # Roadmap is the final node in this phase
    workflow.add_edge("roadmap", END)

    return workflow.compile()


def create_app():
    """Create and return the compiled graph app."""
    return build_graph()
