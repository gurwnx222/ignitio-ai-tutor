"""Critic node for testing user understanding after teaching.

This node:
1. Generates a coding question for each concept taught by the teaching agent
2. Presents the question to the user and collects their code answer
3. Evaluates the user's code for correctness and understanding
4. On failure, sends assessment feedback to the teaching agent for retry
5. Enforces max 1 retry loop per concept

The flow:
- Teaching agent explains concept_1 → Critic tests concept_1
- If fail: Teaching re-explains → Critic tests again (final attempt)
- Then moves to concept_2 (or ends if all concepts tested)
"""

import json
from typing import Optional
from langchain_core.messages import HumanMessage

from core.llm import get_llm
from graph.state import graph_state
from nodes.critic.prompt import (
    QUESTION_GENERATION_PROMPT,
    CODE_EVALUATION_PROMPT,
    ASSESSMENT_SUMMARY_PROMPT,
)


def critic_node(state: graph_state) -> dict:
    """
    Critic node that tests user understanding after concept explanation.

    This node operates in a cycle with the teaching agent:
    1. After teaching explains a concept, critic generates a coding question
    2. User writes code to answer the question
    3. Critic evaluates the code and decides pass/fail
    4. If fail (has_retried=False): send assessment to teaching, return FAIL
       - Teaching will re-explain and set has_retried=True
    5. If fail (has_retried=True): move on with FAIL_FINAL (max 1 retry enforced)

    Note: has_retried is set by the teaching node, not by this node.
    This allows the routing to correctly route to teaching on first failure.

    Args:
        state: Current graph state with explanation, code_examples, concept_map

    Returns:
        dict: Updated state with learning_test results, test_result, and
              assessment_for_teaching if retry is needed
    """
    # Get current state values
    concept_map = state.concept_map
    explanation = state.explanation
    code_examples = state.code_examples
    learning_test = state.learning_test or {}
    has_retried = state.has_retried

    # Determine which concept to test next
    current_concept_index = _get_next_concept_index(learning_test, concept_map)

    # If all concepts tested, we're done
    if current_concept_index is None:
        return {
            "test_result": "COMPLETE",
            "learning_test": learning_test
        }

    # Get the current concept details
    concept_key = f"concept_{current_concept_index}"
    concept_data = concept_map.get(concept_key, {})
    concept_explanation = explanation.get(concept_key, {})
    concept_code = code_examples.get(concept_key, {})

    # Validate we have data to test
    if not concept_data or not concept_code:
        return {
            "test_result": "ERROR",
            "learning_test": learning_test
        }

    # Generate a question for this concept
    question_data = _generate_question(
        concept_name=concept_data.get("name", concept_key),
        concept_description=concept_data.get("description", ""),
        code_example=concept_code.get("code", "")
    )

    # In a real implementation, this would be an interrupt to get user input
    # For now, we simulate user input or use the state's stored answer
    # The actual user interaction happens outside the graph via interrupts
    user_code = _get_user_code_input(state, concept_key)

    # Evaluate the user's code
    evaluation = _evaluate_user_code(
        concept_name=concept_data.get("name", concept_key),
        concept_description=concept_data.get("description", ""),
        question=question_data.get("question", ""),
        code_example=concept_code.get("code", ""),
        user_code=user_code
    )

    # Store the test result
    test_entry = {
        "concept_tested": concept_key,
        "concept_name": concept_data.get("name", concept_key),
        "question": question_data.get("question", ""),
        "user_code": user_code,
        "passed": evaluation.get("passed", False),
        "score": evaluation.get("score", 0),
        "assessment": evaluation.get("assessment", {}),
        "attempt": 2 if has_retried else 1
    }

    # Determine next action based on pass/fail and retry status
    passed = evaluation.get("passed", False)

    if passed:
        # User passed, store the result and move to next concept
        updated_learning_test = {**learning_test}
        updated_learning_test[concept_key] = test_entry
        return {
            "learning_test": updated_learning_test,
            "test_result": "PASS",
            "retry_count": state.retry_count,
            "has_retried": False,  # Reset for next concept
            "assessment_for_teaching": {}  # Clear assessment after pass
        }
    else:
        # User failed
        if not has_retried:
            # First failure - prepare for retry
            # DO NOT store in learning_test yet - concept will be re-tested after teaching
            # DO NOT set has_retried here - teaching will set it when retry runs
            assessment_summary = _create_assessment_summary(
                concept_name=concept_data.get("name", concept_key),
                attempt_number=1,
                passed=False,
                score=evaluation.get("score", 0),
                detailed_assessment=evaluation.get("assessment", {})
            )

            return {
                "learning_test": learning_test,  # Keep original, don't add failed test
                "test_result": "FAIL",
                "retry_count": state.retry_count,  # Don't increment here
                "has_retried": False,  # Keep False so routing goes to teaching
                "assessment_for_teaching": assessment_summary
            }
        else:
            # Second failure - store the final result and move on (max 1 retry)
            updated_learning_test = {**learning_test}
            updated_learning_test[concept_key] = test_entry
            return {
                "learning_test": updated_learning_test,
                "test_result": "FAIL_FINAL",
                "retry_count": state.retry_count,
                "has_retried": True,
                "assessment_for_teaching": {}  # Clear assessment after final fail
            }


def _get_next_concept_index(learning_test: dict, concept_map: dict) -> Optional[int]:
    """
    Determine which concept index to test next.

    Args:
        learning_test: Dict of already-tested concepts
        concept_map: Dict of all concepts

    Returns:
        int or None: Next concept index (1, 2, or 3), or None if all tested
    """
    for i in range(1, 4):
        concept_key = f"concept_{i}"
        if concept_key not in learning_test:
            return i
    return None


def _generate_question(
    concept_name: str,
    concept_description: str,
    code_example: str
) -> dict:
    """
    Generate a coding question for the given concept.

    Args:
        concept_name: Name of the concept to test
        concept_description: Description of the concept
        code_example: The code example the user studied

    Returns:
        dict: Question data including the question itself
    """
    llm = get_llm()

    prompt = QUESTION_GENERATION_PROMPT.format(
        concept_name=concept_name,
        concept_description=concept_description,
        code_example=code_example
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return _parse_json_response(response.content)


def _get_user_code_input(state: graph_state, concept_key: str) -> str:
    """
    Get user's code input from state (set by interrupt mechanism in real app).

    In a production implementation, this would use LangGraph's interrupt
    mechanism to pause execution and wait for user input. The user's
    answer would then be stored in state and retrieved here.

    Args:
        state: Current graph state
        concept_key: The concept being tested

    Returns:
        str: User's code answer
    """
    # Check if user answer is stored in learning_test (from interrupt)
    if state.learning_test and concept_key in state.learning_test:
        stored_test = state.learning_test.get(concept_key, {})
        if "user_code" in stored_test:
            return stored_test["user_code"]

    # Placeholder for demo/testing - in real app, this would come from interrupt
    # Return empty string to indicate no answer provided yet
    return ""


def _evaluate_user_code(
    concept_name: str,
    concept_description: str,
    question: str,
    code_example: str,
    user_code: str
) -> dict:
    """
    Evaluate the user's code answer.

    Args:
        concept_name: Name of the concept being tested
        concept_description: Description of the concept
        question: The question asked
        code_example: The code example they studied
        user_code: The user's code answer

    Returns:
        dict: Evaluation results including passed, score, and assessment
    """
    llm = get_llm()

    prompt = CODE_EVALUATION_PROMPT.format(
        concept_name=concept_name,
        concept_description=concept_description,
        question=question,
        code_example=code_example,
        user_code=user_code if user_code else "(No code provided)"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return _parse_json_response(response.content)


def _create_assessment_summary(
    concept_name: str,
    attempt_number: int,
    passed: bool,
    score: int,
    detailed_assessment: dict
) -> dict:
    """
    Create a summary of the assessment for the teaching agent.

    This summary helps the teaching agent understand what went wrong
    and how to adjust the re-explanation.

    Args:
        concept_name: Name of the concept tested
        attempt_number: Which attempt (1 or 2)
        passed: Whether the user passed
        score: Score achieved
        detailed_assessment: Detailed assessment from evaluation

    Returns:
        dict: Summary for teaching agent
    """
    llm = get_llm()

    prompt = ASSESSMENT_SUMMARY_PROMPT.format(
        concept_name=concept_name,
        attempt_number=attempt_number,
        passed=passed,
        score=score,
        detailed_assessment=json.dumps(detailed_assessment, indent=2)
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return _parse_json_response(response.content)


def _parse_json_response(content: str) -> dict:
    """
    Parse JSON response from LLM, handling formatting issues.

    Args:
        content: Raw LLM response

    Returns:
        dict: Parsed JSON or empty dict on failure
    """
    content = content.strip()

    # Remove markdown code blocks if present
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
        # Return empty structure if parsing fails
        return {}


# Export for LangGraph
__all__ = ["critic_node"]