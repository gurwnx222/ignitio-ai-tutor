from pydantic import BaseModel, Field
from typing import Optional


class graph_state(BaseModel):
    """State schema for the ignitio-ai-tutor LangGraph pipeline.

    The pipeline follows: Orchestrator → Builder → Teaching → Critic
    With a single reflective loop: Critic → Teaching (on failure, max once)
    """

    # User input
    user_query: str = Field(..., description="The user's meme generation request")

    # Orchestrator output
    sub_tasks: dict = Field(default_factory=dict, description="Sub-tasks created by orchestrator for downstream agents")

    # Builder output
    meme_url: str = Field(default="", description="URL of the generated meme image")
    meme_text: str = Field(default="", description="Text used in the meme (top/bottom)")
    concept_map: dict = Field(default_factory=dict, description="3 core LangChain concepts used in the meme generator")

    # Teaching output
    explanation: dict = Field(default_factory=dict, description="First-principles + Feynman explanations for each concept")
    code_examples: dict = Field(default_factory=dict, description="Real-world code examples for each concept")

    # Critic output
    learning_test: dict = Field(default_factory=dict, description="Learning test questions and user responses")
    test_result: str = Field(default="", description="PASS, FAIL, FAIL_FINAL, or COMPLETE result from critic evaluation")

    # Assessment feedback for teaching agent (sent after critic evaluates user code)
    assessment_for_teaching: dict = Field(default_factory=dict, description="Assessment and score sent to teaching agent when user fails test")

    # Reflective loop tracking (max 1 retry)
    retry_count: int = Field(default=0, description="Number of teaching retries attempted (max 1)")
    has_retried: bool = Field(default=False, description="Whether the reflective loop has already executed")