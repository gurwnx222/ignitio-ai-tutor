"""Custom exceptions for the FastAPI application."""

from fastapi import HTTPException, status


class InvalidMemeRequestError(HTTPException):
    """Raised when the user query is not a valid meme request."""

    def __init__(self, detail: str = "Invalid meme request. Please provide a meme-related query."):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )


class SessionNotFoundError(HTTPException):
    """Raised when a session ID is not found in the session store."""

    def __init__(self, session_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )


class GraphExecutionError(HTTPException):
    """Raised when the LangGraph pipeline encounters an error."""

    def __init__(self, detail: str = "An error occurred while processing your request."):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


class RateLimitError(HTTPException):
    """Raised when the imgflip API rate limit is exceeded."""

    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Please try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )


class SessionExpiredError(HTTPException):
    """Raised when a session has expired."""

    def __init__(self, session_id: str):
        super().__init__(
            status_code=status.HTTP_410_GONE,
            detail=f"Session {session_id} has expired. Please start a new session."
        )


class InvalidSessionStateError(HTTPException):
    """Raised when the session is in an invalid state for the requested operation."""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )