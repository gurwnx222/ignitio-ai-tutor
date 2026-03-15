"""Pydantic models for API request and response schemas."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Request Models
# ============================================================================

class MemeRequest(BaseModel):
    """Request to start a new meme generation and learning session."""

    user_query: str = Field(
        ...,
        description="The user's meme request, e.g., 'Create a meme about debugging code at 3am'",
        min_length=5,
        max_length=500
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_query": "Create a meme about debugging code at 3am"
            }
        }


class TestAnswerRequest(BaseModel):
    """Request to submit test answers for evaluation."""

    session_id: str = Field(
        ...,
        description="The session ID from the initial meme request"
    )
    answers: Dict[str, str] = Field(
        ...,
        description="Dictionary mapping concept keys to user's code answers",
        examples=[{
            "concept_1": "# My answer for concept 1\nprint('Hello World')",
            "concept_2": "# My answer for concept 2\ndef my_function(): pass",
            "concept_3": "# My answer for concept 3\nreturn True"
        }]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "answers": {
                    "concept_1": "def hello(): print('world')",
                    "concept_2": "from langchain import LLMChain",
                    "concept_3": "chain = LLMChain(llm=llm, prompt=prompt)"
                }
            }
        }


# ============================================================================
# Response Models
# ============================================================================

class ConceptInfo(BaseModel):
    """Information about a single LangChain concept."""

    name: str = Field(..., description="The name of the concept")
    description: str = Field(..., description="Brief description of the concept")


class MemeResponse(BaseModel):
    """Response after successful meme generation and concept explanation."""

    session_id: str = Field(..., description="Unique session ID for this interaction")
    meme_url: str = Field(..., description="URL to the generated meme image")
    meme_text: str = Field(..., description="Text used in the meme (top/bottom)")
    concept_map: Dict[str, ConceptInfo] = Field(
        ...,
        description="3 core LangChain concepts used in the meme generator"
    )
    explanation: Dict = Field(
        ...,
        description="Educational explanations for each concept"
    )
    code_examples: Dict = Field(
        ...,
        description="Code examples for each concept"
    )
    current_step: str = Field(
        ...,
        description="Current step in the pipeline (init, teaching, test_ready, completed)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "meme_url": "https://i.imgflip.com/abc123.jpg",
                "meme_text": "Debugging at 3am | It works on my machine",
                "concept_map": {
                    "concept_1": {
                        "name": "Language Models",
                        "description": "AI models that understand and generate text"
                    },
                    "concept_2": {
                        "name": "Prompt Engineering",
                        "description": "Crafting effective instructions for AI"
                    },
                    "concept_3": {
                        "name": "Chains",
                        "description": "Sequencing multiple AI operations"
                    }
                },
                "explanation": {
                    "concept_1": {
                        "name": "Language Models",
                        "introduction": "...",
                        "feynman_explanation": "...",
                        "analogy": "..."
                    }
                },
                "code_examples": {
                    "concept_1": {
                        "code": "...",
                        "explanation": "..."
                    }
                },
                "current_step": "test_ready"
            }
        }


class TestQuestion(BaseModel):
    """A single test question for a concept."""

    question: str = Field(..., description="The coding question to answer")
    concept_name: str = Field(..., description="Name of the concept being tested")
    concept_key: str = Field(..., description="Key identifier (concept_1, concept_2, concept_3)")


class TestQuestionResponse(BaseModel):
    """Response with test questions for the user."""

    session_id: str = Field(..., description="Session ID")
    questions: List[TestQuestion] = Field(
        ...,
        description="List of test questions, one per concept"
    )
    instructions: str = Field(
        ...,
        description="Instructions for the user on how to answer"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "questions": [
                    {
                        "question": "Write code to initialize a simple LLM using LangChain",
                        "concept_name": "Language Models",
                        "concept_key": "concept_1"
                    },
                    {
                        "question": "Create a prompt template for generating memes",
                        "concept_name": "Prompt Engineering",
                        "concept_key": "concept_2"
                    },
                    {
                        "question": "Chain the LLM and prompt together",
                        "concept_name": "Chains",
                        "concept_key": "concept_3"
                    }
                ],
                "instructions": "Please answer each question with working code. You can reference the code examples provided."
            }
        }


class TestResultResponse(BaseModel):
    """Response after evaluating user's test answers."""

    session_id: str = Field(..., description="Session ID")
    passed: bool = Field(..., description="Whether the user passed all tests")
    test_result: str = Field(
        ...,
        description="Result status: PASS, FAIL, FAIL_FINAL, or COMPLETE"
    )
    results: Dict[str, Dict] = Field(
        ...,
        description="Detailed results for each concept test"
    )
    feedback: Optional[str] = Field(
        None,
        description="Overall feedback on performance"
    )
    retry_available: bool = Field(
        ...,
        description="Whether a retry is available (only 1 retry allowed)"
    )
    new_explanation: Optional[Dict] = Field(
        None,
        description="Simplified explanation if retry is triggered"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "passed": False,
                "test_result": "FAIL",
                "results": {
                    "concept_1": {
                        "passed": True,
                        "score": 85,
                        "feedback": "Good understanding of LLM initialization"
                    },
                    "concept_2": {
                        "passed": False,
                        "score": 40,
                        "feedback": "Try reviewing prompt templates again"
                    }
                },
                "feedback": "You passed 1 out of 3 concepts. Let's try again with simpler explanations.",
                "retry_available": True,
                "new_explanation": {
                    "concept_2": {
                        "simplified_explanation": "..."
                    }
                }
            }
        }


class SessionResponse(BaseModel):
    """Response with current session state."""

    session_id: str = Field(..., description="Session ID")
    current_step: str = Field(..., description="Current pipeline step")
    user_query: str = Field(..., description="Original user query")
    meme_url: Optional[str] = Field(None, description="Generated meme URL")
    concept_map: Optional[Dict] = Field(None, description="Concepts identified")
    explanation: Optional[Dict] = Field(None, description="Concept explanations")
    code_examples: Optional[Dict] = Field(None, description="Code examples")
    test_result: Optional[str] = Field(None, description="Test result if completed")
    created_at: str = Field(..., description="Session creation timestamp")
    expires_at: str = Field(..., description="Session expiration timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "current_step": "test_ready",
                "user_query": "Create a meme about debugging code at 3am",
                "meme_url": "https://i.imgflip.com/abc123.jpg",
                "concept_map": {
                    "concept_1": {"name": "Language Models", "description": "..."}
                },
                "explanation": {"concept_1": {"..."}},
                "code_examples": {"concept_1": {"..."}},
                "test_result": None,
                "created_at": "2024-01-15T10:30:00",
                "expires_at": "2024-01-15T11:30:00"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict] = Field(None, description="Additional error details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "InvalidMemeRequest",
                "message": "The query is not related to meme generation",
                "details": {"reasoning": "The query asks about weather, not memes"}
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Current environment")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "environment": "development"
            }
        }