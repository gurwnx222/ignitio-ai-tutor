"""Tutor API routes for the ignitio-ai-tutor pipeline.

Endpoints:
- POST /api/v1/tutor/start - Start a new session with meme generation
- GET /api/v1/tutor/session/{session_id} - Get session state
- POST /api/v1/tutor/test/start - Get test questions
- POST /api/v1/tutor/test/submit - Submit test answers
"""

from typing import Dict
from fastapi import APIRouter, HTTPException, status

from graph.state import graph_state
from graph.builder import create_partial_app, run_teaching_retry
from api.sessions import SessionManager, get_session_manager, Session
from api.schemas import (
    MemeRequest,
    MemeResponse,
    TestQuestionResponse,
    TestAnswerRequest,
    TestResultResponse,
    SessionResponse,
    ErrorResponse,
    TestQuestion,
    ConceptInfo,
)
from api.exceptions import (
    InvalidMemeRequestError,
    SessionNotFoundError,
    SessionExpiredError,
    InvalidSessionStateError,
    GraphExecutionError,
)
from api.config import settings

router = APIRouter(prefix="/tutor", tags=["tutor"])

# LangGraph application instance (compiled once)
_partial_app = None


def get_partial_app():
    """Get or create the partial LangGraph application instance (without critic)."""
    global _partial_app
    if _partial_app is None:
        _partial_app = create_partial_app()
    return _partial_app


@router.post(
    "/start",
    response_model=MemeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid meme request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Start a new learning session",
    description="Submit a meme request to start a new learning session. Returns meme, concepts, and explanations."
)
async def start_session(request: MemeRequest) -> MemeResponse:
    """
    Start a new learning session.

    This endpoint:
    1. Validates the user's meme request
    2. Generates a meme via imgflip API
    3. Identifies 3 LangChain concepts
    4. Provides explanations and code examples
    5. Stores session state for subsequent test questions

    Returns session_id for use in test endpoints.
    """
    try:
        # Create initial state
        initial_state = graph_state(user_query=request.user_query)

        # Get the partial app (orchestrator → builder → teaching)
        app = get_partial_app()

        # Execute orchestrator → builder → teaching pipeline
        result = app.invoke(initial_state)

        # Check if orchestrator rejected the query
        sub_tasks = result.get("sub_tasks", {})
        if not sub_tasks.get("is_valid", False):
            raise InvalidMemeRequestError(
                detail=sub_tasks.get("reasoning", "Query is not a valid meme request")
            )

        # Create session with the result
        session_manager = get_session_manager()

        # Create a proper graph_state from the result
        state = graph_state(**result)

        # Create session with the state
        session = session_manager.create_session(state)

        # Update session step to test_ready
        session_manager.update_session(
            session_id=session.id,
            state=state,
            current_step="test_ready"
        )

        # Build concept map for response
        concept_map = {}
        for key, concept in result.get("concept_map", {}).items():
            concept_map[key] = ConceptInfo(
                name=concept.get("name", key),
                description=concept.get("description", "")
            )

        return MemeResponse(
            session_id=session.id,
            meme_url=result.get("meme_url", ""),
            meme_text=result.get("meme_text", ""),
            concept_map=concept_map,
            explanation=result.get("explanation", {}),
            code_examples=result.get("code_examples", {}),
            current_step="test_ready"
        )

    except InvalidMemeRequestError:
        raise
    except Exception as e:
        raise GraphExecutionError(detail=f"Failed to process request: {str(e)}")


@router.get(
    "/session/{session_id}",
    response_model=SessionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
        410: {"model": ErrorResponse, "description": "Session expired"}
    },
    summary="Get session state",
    description="Retrieve the current state of a learning session."
)
async def get_session(session_id: str) -> SessionResponse:
    """
    Get the current state of a session.

    Returns all session data including meme, concepts, explanations, and test results.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if session is None:
        raise SessionNotFoundError(session_id)

    if session.is_expired():
        raise SessionExpiredError(session_id)

    state = session.state

    return SessionResponse(
        session_id=session.id,
        current_step=session.current_step,
        user_query=state.user_query,
        meme_url=state.meme_url if state.meme_url else None,
        concept_map=state.concept_map if state.concept_map else None,
        explanation=state.explanation if state.explanation else None,
        code_examples=state.code_examples if state.code_examples else None,
        test_result=state.test_result if state.test_result else None,
        created_at=session.created_at.isoformat(),
        expires_at=session.expires_at.isoformat()
    )


@router.post(
    "/test/start",
    response_model=TestQuestionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
        400: {"model": ErrorResponse, "description": "Invalid session state"}
    },
    summary="Get test questions",
    description="Generate test questions for the concepts in the session."
)
async def start_test(session_id: str) -> TestQuestionResponse:
    """
    Start the learning test for a session.

    Generates 3 coding questions, one for each concept.
    User should answer these and submit via /test/submit.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if session is None:
        raise SessionNotFoundError(session_id)

    if session.is_expired():
        raise SessionExpiredError(session_id)

    state = session.state

    # Validate session is in correct state
    if session.current_step not in ("test_ready", "teaching"):
        raise InvalidSessionStateError(
            f"Session is in '{session.current_step}' state. Must be in 'test_ready' state to start test."
        )

    # Get concept map
    concept_map = state.concept_map
    if not concept_map:
        raise InvalidSessionStateError("No concepts found in session. Start a new session first.")

    # Generate test questions
    questions = []
    for i in range(1, 4):
        concept_key = f"concept_{i}"
        concept_data = concept_map.get(concept_key, {})
        if concept_data:
            # Generate a question for this concept using LLM
            question_text = _generate_test_question(
                concept_name=concept_data.get("name", concept_key),
                concept_description=concept_data.get("description", ""),
                code_example=state.code_examples.get(concept_key, {}).get("code", "")
            )
            questions.append(TestQuestion(
                question=question_text,
                concept_name=concept_data.get("name", concept_key),
                concept_key=concept_key
            ))

    # Store questions in session
    questions_dict = {q.concept_key: {"question": q.question, "concept_name": q.concept_name} for q in questions}
    session_manager.set_test_questions(session_id, questions_dict)

    # Update session step
    session_manager.update_session(session_id, state, current_step="testing")

    return TestQuestionResponse(
        session_id=session_id,
        questions=questions,
        instructions="Please answer each question with working code. You can reference the code examples provided earlier."
    )


@router.post(
    "/test/submit",
    response_model=TestResultResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
        400: {"model": ErrorResponse, "description": "Invalid session state"}
    },
    summary="Submit test answers",
    description="Submit user's code answers for evaluation."
)
async def submit_test(request: TestAnswerRequest) -> TestResultResponse:
    """
    Submit test answers for evaluation.

    Evaluates each answer and returns pass/fail result.
    If failed and retry available, includes new simplified explanation.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(request.session_id)

    if session is None:
        raise SessionNotFoundError(request.session_id)

    if session.is_expired():
        raise SessionExpiredError(request.session_id)

    state = session.state

    # Validate session is in testing state
    if session.current_step != "testing":
        raise InvalidSessionStateError(
            f"Session is in '{session.current_step}' state. Must start test first."
        )

    # Get stored test questions
    test_questions = session_manager.get_test_questions(request.session_id)
    if not test_questions:
        raise InvalidSessionStateError("No test questions found for this session. Start the test first.")

    # Evaluate each answer
    results = {}
    all_passed = True

    for concept_key, user_code in request.answers.items():
        concept_data = state.concept_map.get(concept_key, {})
        concept_name = concept_data.get("name", concept_key)
        question_data = test_questions.get(concept_key, {})

        # Evaluate the code answer
        evaluation = _evaluate_code_answer(
            concept_name=concept_name,
            concept_description=concept_data.get("description", ""),
            question=question_data.get("question", ""),
            code_example=state.code_examples.get(concept_key, {}).get("code", ""),
            user_code=user_code
        )

        results[concept_key] = {
            "passed": evaluation.get("passed", False),
            "score": evaluation.get("score", 0),
            "feedback": evaluation.get("feedback", ""),
            "concept_name": concept_name
        }

        if not evaluation.get("passed", False):
            all_passed = False

    # Determine overall result
    has_retried = state.has_retried

    if all_passed:
        test_result = "PASS"
        retry_available = False
        new_explanation = None
    elif has_retried:
        test_result = "FAIL_FINAL"
        retry_available = False
        new_explanation = None
    else:
        test_result = "FAIL"
        retry_available = True

        # Generate simpler explanation for failed concepts
        failed_concepts = [k for k, v in results.items() if not v["passed"]]
        new_explanation = _generate_simpler_explanation(
            state=state,
            failed_concepts=failed_concepts
        )

        # Update state for retry (but don't set has_retried yet - that happens after retry submission)
        state.test_result = test_result
        state.assessment_for_teaching = {
            "key_misunderstandings": [
                {"concept": results[k]["concept_name"], "issues": results[k]["feedback"]}
                for k in failed_concepts
            ],
            "suggested_focus": "Focus on simpler explanations with more analogies"
        }
        session_manager.update_session(
            session_id=request.session_id,
            state=state,
            current_step="teaching"
        )

    # If passed or final fail, mark as complete
    if test_result in ("PASS", "FAIL_FINAL"):
        session_manager.update_session(
            session_id=request.session_id,
            state=state,
            current_step="completed"
        )
        state.test_result = test_result

    # Build feedback message
    passed_count = sum(1 for v in results.values() if v["passed"])
    feedback = f"You passed {passed_count} out of {len(results)} concepts."

    if test_result == "PASS":
        feedback += " Great job! You've completed this learning session."
    elif test_result == "FAIL":
        feedback += " Let's try again with simpler explanations. Use the /test/retry endpoint to get new questions."
    else:
        feedback += " Session complete. Review the concepts and try again later."

    return TestResultResponse(
        session_id=request.session_id,
        passed=all_passed,
        test_result=test_result,
        results=results,
        feedback=feedback,
        retry_available=retry_available,
        new_explanation=new_explanation
    )


@router.post(
    "/test/retry",
    response_model=TestQuestionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
        400: {"model": ErrorResponse, "description": "Invalid session state"}
    },
    summary="Retry test with simpler explanations",
    description="Get new test questions after receiving simpler explanations (only available after first test failure)."
)
async def retry_test(session_id: str) -> TestQuestionResponse:
    """
    Retry the test with simpler explanations.

    This endpoint is only available after a first test failure.
    It provides simpler explanations and new test questions.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if session is None:
        raise SessionNotFoundError(session_id)

    if session.is_expired():
        raise SessionExpiredError(session_id)

    state = session.state

    # Validate session is in teaching state (ready for retry)
    if session.current_step != "teaching":
        raise InvalidSessionStateError(
            f"Session is in '{session.current_step}' state. Retry is only available after a test failure."
        )

    # Verify user hasn't already retried
    if state.has_retried:
        raise InvalidSessionStateError(
            "You have already used your retry. Session is complete."
        )

    # Get concept map
    concept_map = state.concept_map
    if not concept_map:
        raise InvalidSessionStateError("No concepts found in session.")

    # Generate new test questions
    questions = []
    for i in range(1, 4):
        concept_key = f"concept_{i}"
        concept_data = concept_map.get(concept_key, {})
        if concept_data:
            question_text = _generate_test_question(
                concept_name=concept_data.get("name", concept_key),
                concept_description=concept_data.get("description", ""),
                code_example=state.code_examples.get(concept_key, {}).get("code", "")
            )
            questions.append(TestQuestion(
                question=question_text,
                concept_name=concept_data.get("name", concept_key),
                concept_key=concept_key
            ))

    # Store questions in session
    questions_dict = {q.concept_key: {"question": q.question, "concept_name": q.concept_name} for q in questions}
    session_manager.set_test_questions(session_id, questions_dict)

    # Mark that retry has been used
    state.has_retried = True

    # Update session step to testing
    session_manager.update_session(session_id, state, current_step="testing")

    # Build instructions with simpler explanation if available
    instructions = "Here are the test questions again. You have simpler explanations available. Good luck!"
    if state.assessment_for_teaching:
        # The user has access to the new_explanation from the previous submit response
        instructions += " Review the simplified explanations provided earlier."

    return TestQuestionResponse(
        session_id=session_id,
        questions=questions,
        instructions=instructions
    )


# ============================================================================
# Helper Functions
# ============================================================================

def _generate_test_question(
    concept_name: str,
    concept_description: str,
    code_example: str
) -> str:
    """
    Generate a test question for a concept using the LLM.

    Uses the critic node's question generation logic.
    """
    from nodes.critic.node import _generate_question

    question_data = _generate_question(
        concept_name=concept_name,
        concept_description=concept_description,
        code_example=code_example
    )

    return question_data.get("question", f"Write code to demonstrate {concept_name}")


def _evaluate_code_answer(
    concept_name: str,
    concept_description: str,
    question: str,
    code_example: str,
    user_code: str
) -> Dict:
    """
    Evaluate user's code answer using the LLM.

    Uses the critic node's evaluation logic.
    """
    from nodes.critic.node import _evaluate_user_code

    return _evaluate_user_code(
        concept_name=concept_name,
        concept_description=concept_description,
        question=question,
        code_example=code_example,
        user_code=user_code
    )


def _generate_simpler_explanation(
    state: graph_state,
    failed_concepts: list
) -> Dict:
    """
    Generate simpler explanations for failed concepts.

    Uses the teaching node's simpler explanation logic.
    """
    from nodes.teaching.node import _generate_simpler_explanation as _gen_simpler

    # Format concepts for the prompt
    concepts_formatted = ""
    for key, concept in state.concept_map.items():
        concepts_formatted += f"- {concept.get('name', key)}: {concept.get('description', '')}\n"

    return _gen_simpler(concepts_formatted, state.meme_text)