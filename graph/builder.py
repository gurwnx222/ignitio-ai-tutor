"""LangGraph builder for the ignitio-ai-tutor pipeline.

This module creates and compiles the LangGraph workflow that connects:
    Orchestrator → Builder → Teaching → Critic

With a conditional reflective loop:
    Critic → Teaching (on FAIL, max 1 retry)

The graph uses graph_state from graph/state.py as its state schema.
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from graph.state import graph_state
from nodes.orchestrator.node import orchestrator_node
from nodes.builder.node import builder_node
from nodes.teaching.node import teaching_node
from nodes.critic.node import critic_node


def should_continue_after_orchestrator(state: graph_state) -> Literal["builder", "end"]:
    """
    Route after orchestrator validates the meme request.

    Args:
        state: Current graph state

    Returns:
        "builder" if valid meme request, "end" otherwise
    """
    sub_tasks = state.sub_tasks

    # Check if the query was validated as a meme request
    if not sub_tasks or not sub_tasks.get("is_valid", False):
        return "end"

    return "builder"


def should_continue_after_critic(state: graph_state) -> Literal["teaching", "end"]:
    """
    Route after critic evaluates the learning test.

    Routing logic:
    - PASS or COMPLETE: User passed, end session
    - FAIL: First failure, retry teaching with simpler explanation
    - FAIL_FINAL: Second failure, end session (max 1 retry enforced)

    Args:
        state: Current graph state

    Returns:
        "teaching" if retry needed, "end" otherwise
    """
    test_result = state.test_result
    has_retried = state.has_retried

    # If test passed, we're done
    if test_result in ("PASS", "COMPLETE"):
        return "end"

    # If this is the first failure, go back to teaching
    if test_result == "FAIL" and not has_retried:
        return "teaching"

    # FAIL_FINAL or any other result ends the session
    return "end"


def create_app():
    """
    Create and compile the full LangGraph application.

    The graph topology:

        START → orchestrator → [builder → teaching → critic] → END
                           │                      │
                           │                      └──(on FAIL)→ teaching (retry)
                           │                              │
                           └──(if invalid)→ END           └──→ critic → END

    Returns:
        Compiled LangGraph application
    """
    # Create the graph with our state schema
    workflow = StateGraph(graph_state)

    # Add all nodes to the graph
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("builder", builder_node)
    workflow.add_node("teaching", teaching_node)
    workflow.add_node("critic", critic_node)

    # Set the entry point
    workflow.set_entry_point("orchestrator")

    # Add edges
    # Orchestrator → Builder (conditional) or END
    workflow.add_conditional_edges(
        "orchestrator",
        should_continue_after_orchestrator,
        {
            "builder": "builder",
            "end": END
        }
    )

    # Builder → Teaching (always)
    workflow.add_edge("builder", "teaching")

    # Teaching → Critic (always)
    workflow.add_edge("teaching", "critic")

    # Critic → Teaching (on retry) or END (on pass/final fail)
    workflow.add_conditional_edges(
        "critic",
        should_continue_after_critic,
        {
            "teaching": "teaching",
            "end": END
        }
    )

    # Compile the graph
    app = workflow.compile()
    return app


def create_partial_app():
    """
    Create and compile a partial graph for the teaching pipeline only.

    This is used by the API to run only:
        Orchestrator → Builder → Teaching

    The critic is handled separately via direct function calls
    to allow for user interaction between teaching and testing.

    Returns:
        Compiled partial LangGraph application (without critic)
    """
    workflow = StateGraph(graph_state)

    # Add nodes for partial pipeline
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("builder", builder_node)
    workflow.add_node("teaching", teaching_node)

    # Set entry point
    workflow.set_entry_point("orchestrator")

    # Add conditional edge from orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        should_continue_after_orchestrator,
        {
            "builder": "builder",
            "end": END
        }
    )

    # Builder → Teaching → END
    workflow.add_edge("builder", "teaching")
    workflow.add_edge("teaching", END)

    return workflow.compile()


def run_teaching_retry(state: graph_state) -> dict:
    """
    Run the teaching node for retry after test failure.

    This is used when the user fails the test and needs
    simpler explanations. The state should have:
    - assessment_for_teaching: feedback from critic
    - has_retried: should be False (will be set to True)

    Args:
        state: Current graph state with assessment_for_teaching

    Returns:
        dict: Updated state with new explanation and code_examples
    """
    return teaching_node(state)


# For direct imports
__all__ = ["create_app", "create_partial_app", "run_teaching_retry", "should_continue_after_orchestrator", "should_continue_after_critic"]