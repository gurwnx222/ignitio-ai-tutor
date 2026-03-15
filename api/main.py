"""FastAPI application entry point for ignitio-ai-tutor.

This module creates and configures the FastAPI application with:
- CORS middleware for cross-origin requests
- Exception handlers for custom errors
- Health check endpoint
- Tutor API routes
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.routes import tutor_router
from api.schemas import ErrorResponse, HealthResponse
from api.exceptions import (
    InvalidMemeRequestError,
    SessionNotFoundError,
    SessionExpiredError,
    InvalidSessionStateError,
    GraphExecutionError,
    RateLimitError,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    - Initialize session manager on startup
    - Cleanup on shutdown
    """
    # Startup: Initialize session manager
    from api.sessions import get_session_manager
    get_session_manager()
    print(f"Starting {settings.api_title} in {settings.environment} mode")

    yield

    # Shutdown: Cleanup
    print("Shutting down...")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="""
Ignitio AI Tutor API - A LangGraph-based learning assistant.

## Overview

This API provides endpoints for:
- Starting a new learning session with meme generation
- Retrieving session state
- Taking learning tests
- Submitting test answers for evaluation

## Flow

1. **POST /api/v1/tutor/start** - Submit a meme request, get back meme + concepts + explanations
2. **POST /api/v1/tutor/test/start** - Get test questions for the concepts
3. **POST /api/v1/tutor/test/submit** - Submit your answers for evaluation
4. **GET /api/v1/tutor/session/{id}** - Check session state at any time

## Retry Policy

- If you fail the test, you get one retry with simplified explanations
- After the retry, the session is complete regardless of result
    """,
    lifespan=lifespan,
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(InvalidMemeRequestError)
async def invalid_meme_request_handler(request: Request, exc: InvalidMemeRequestError):
    """Handle invalid meme request errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="InvalidMemeRequest",
            message=exc.detail,
            details=None
        ).model_dump()
    )


@app.exception_handler(SessionNotFoundError)
async def session_not_found_handler(request: Request, exc: SessionNotFoundError):
    """Handle session not found errors."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=ErrorResponse(
            error="SessionNotFound",
            message=exc.detail,
            details=None
        ).model_dump()
    )


@app.exception_handler(SessionExpiredError)
async def session_expired_handler(request: Request, exc: SessionExpiredError):
    """Handle session expired errors."""
    return JSONResponse(
        status_code=status.HTTP_410_GONE,
        content=ErrorResponse(
            error="SessionExpired",
            message=exc.detail,
            details=None
        ).model_dump()
    )


@app.exception_handler(InvalidSessionStateError)
async def invalid_session_state_handler(request: Request, exc: InvalidSessionStateError):
    """Handle invalid session state errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="InvalidSessionState",
            message=exc.detail,
            details=None
        ).model_dump()
    )


@app.exception_handler(GraphExecutionError)
async def graph_execution_error_handler(request: Request, exc: GraphExecutionError):
    """Handle graph execution errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="GraphExecutionError",
            message=exc.detail,
            details=None
        ).model_dump()
    )


@app.exception_handler(RateLimitError)
async def rate_limit_handler(request: Request, exc: RateLimitError):
    """Handle rate limit errors."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=ErrorResponse(
            error="RateLimitExceeded",
            message=exc.detail,
            details={"retry_after": exc.headers.get("Retry-After", "60")}
        ).model_dump(),
        headers=exc.headers
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    # Log the actual error for debugging
    print(f"Unexpected error: {type(exc).__name__}: {str(exc)}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred. Please try again later.",
            details={"error_type": type(exc).__name__} if settings.environment == "development" else None
        ).model_dump()
    )


# ============================================================================
# Routes
# ============================================================================

# Include tutor routes
app.include_router(tutor_router, prefix=settings.api_prefix)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
    description="Check if the API is running and healthy."
)
async def health_check():
    """
    Health check endpoint.

    Returns basic information about the API status.
    """
    from api.sessions import get_session_manager

    session_manager = get_session_manager()
    active_sessions = session_manager.get_session_count()

    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        environment=settings.environment
    )


@app.get(
    "/",
    tags=["root"],
    summary="Root endpoint",
    description="API root - redirects to documentation."
)
async def root():
    """
    Root endpoint.

    Returns basic API information.
    """
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


# Export app for uvicorn
__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development"
    )