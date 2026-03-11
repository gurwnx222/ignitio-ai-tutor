"""Orchestrator node for validating meme requests and creating sub-tasks.

This node:
1. Validates if the user query is a meme-related request
2. If valid, creates structured sub-tasks for Builder, Teaching, and Critic agents
3. Returns the sub-tasks to be dispatched through the LangGraph pipeline
"""

import json
from typing import TypedDict, cast
from langchain_core.messages import HumanMessage

from core.llm import get_llm
from graph.state import graph_state
from nodes.orchestrator.prompt import (
    MEME_VALIDATION_PROMPT,
    MEME_PLAN_PROMPT,
    NON_MEME_RESPONSE,
)


class ValidationResult(TypedDict):
    """Schema for meme validation response."""
    is_meme_request: bool
    reasoning: str


class SubTasks(TypedDict):
    """Schema for sub-tasks created by orchestrator."""
    builder_task: dict
    teaching_task: dict
    critic_task: dict


def orchestrator_node(state: graph_state) -> dict:
    """
    Orchestrator node that validates user input and creates sub-tasks.

    This node performs two main functions:
    1. Validates whether the user query is a meme generation request
    2. If valid, creates a structured task plan for downstream agents

    Args:
        state: Current graph state containing user_query

    Returns:
        dict: Updated state with sub_tasks or error response for non-meme queries
    """
    llm = get_llm()
    user_query = state.user_query

    # Step 1: Validate if the query is a meme request
    validation_prompt = MEME_VALIDATION_PROMPT.format(user_query=user_query)
    validation_response = llm.invoke([HumanMessage(content=validation_prompt)])

    # Parse validation result
    validation_result = _parse_json_response(validation_response.content, ValidationResult)

    # If not a meme request, return early with non-meme response
    if not validation_result.get("is_meme_request", False):
        return {
            "sub_tasks": {
                "is_valid": False,
                "response": NON_MEME_RESPONSE,
                "reasoning": validation_result.get("reasoning", "Not a meme request")
            }
        }

    # Step 2: Create sub-tasks for the agent pipeline
    plan_prompt = MEME_PLAN_PROMPT.format(user_query=user_query)
    plan_response = llm.invoke([HumanMessage(content=plan_prompt)])

    # Parse the task plan
    task_plan = _parse_json_response(plan_response.content, dict)

    # Structure the sub-tasks for downstream agents
    sub_tasks = _create_sub_tasks(task_plan)

    return {"sub_tasks": sub_tasks}


def _parse_json_response(response_content: str, schema: type) -> dict:
    """
    Parse JSON response from LLM, handling potential formatting issues.

    Args:
        response_content: Raw string response from LLM
        schema: Expected schema type for validation

    Returns:
        dict: Parsed JSON object
    """
    # Clean up response - remove markdown code blocks if present
    content = response_content.strip()

    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]

    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Return a default structure if parsing fails
        if schema == ValidationResult:
            return {"is_meme_request": False, "reasoning": "Failed to parse validation response"}
        return {}


def _create_sub_tasks(task_plan: dict) -> dict:
    """
    Structure the task plan into sub-tasks for each agent.

    The sub-tasks follow this structure:
    - Builder: generates meme_url, meme_text, concept_map (3 LangChain concepts)
    - Teaching: explains concepts using first principles + Feynman technique,
                provides real-world code examples for the meme generator
    - Critic: tests user understanding, handles reflective loop on failure

    Args:
        task_plan: Raw task plan from LLM

    Returns:
        dict: Structured sub-tasks for the pipeline
    """
    return {
        "is_valid": True,
        "builder_task": {
            "description": "Generate meme assets and identify core LangChain concepts",
            "meme_url": task_plan.get("builder_task", {}).get("meme_url", ""),
            "meme_text": task_plan.get("builder_task", {}).get("meme_text", ""),
            "concept_map": task_plan.get("builder_task", {}).get("concept_map", []),
        },
        "teaching_task": {
            "description": "Explain concepts using first principles and Feynman technique",
            "explanation_approach": {
                "method": "first_principles_feynman",
                "principles": [
                    "Break down each concept into its fundamental components",
                    "Use simple language and analogies",
                    "Connect concepts directly to the generated meme",
                    "Present information incrementally to avoid cognitive overload",
                    "Use the Feynman technique: explain as if teaching a beginner"
                ],
                "concepts": task_plan.get("teaching_tasks", {}),
            },
            "code_examples": {
                "description": "Concrete real-world code examples for the meme generator",
                "examples": task_plan.get("teaching_tasks", {}),
            }
        },
        "critic_task": {
            "description": "Test user understanding and manage reflective loop",
            "learning_test": {
                "questions": task_plan.get("critic_task", {}).get("learning_test", []),
                "pass_criteria": "User demonstrates understanding of all 3 concepts",
            },
            "reflective_loop": {
                "max_retries": 1,
                "on_fail": {
                    "action": "return_to_teaching",
                    "instruction": "Provide simpler, more digestible explanations using analogies and visual aids",
                    "focus_areas": ["concepts_user_struggled_with"]
                }
            }
        }
    }


# Export for LangGraph
__all__ = ["orchestrator_node"]