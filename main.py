"""
Entry point for ignitio-ai-tutor.

Provides both CLI and API server modes:
- CLI mode: Run the pipeline directly with a sample query
- API mode: Start FastAPI server for REST endpoints

Usage:
    # CLI mode (default)
    uv run python main.py

    # API server mode
    uv run python main.py --api

    # API server with custom host/port
    uv run python main.py --api --host 0.0.0.0 --port 8080
"""

import argparse
import sys


def run_cli():
    """Run the pipeline in CLI mode with a sample query."""
    from graph.builder import create_app
    from graph.state import graph_state

    print("Running ignitio-ai-tutor in CLI mode...")
    print("=" * 60)

    app = create_app()

    # Create initial state with a meme request
    initial_state = graph_state(
        user_query="Create a meme about debugging code at 3am"
    )

    # Invoke the graph
    result = app.invoke(initial_state)

    # Print the final result
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"User Query: {result.get('user_query', 'N/A')}")
    print(f"Meme URL: {result.get('meme_url', 'N/A')}")
    print(f"Meme Text: {result.get('meme_text', 'N/A')}")
    print(f"Concept Map: {result.get('concept_map', {})}")
    print(f"Test Result: {result.get('test_result', 'N/A')}")
    print("=" * 60)

    return result


def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server.

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8000)
        reload: Enable auto-reload for development (default: False)
    """
    import uvicorn
    from api.config import settings

    print(f"Starting {settings.api_title}...")
    print(f"Environment: {settings.environment}")
    print(f"Server: http://{host}:{port}")
    print(f"Docs: http://{host}:{port}/docs")
    print(f"OpenAPI: http://{host}:{port}/openapi.json")
    print("=" * 60)

    # Use reload in development mode if not explicitly set
    if reload is None:
        reload = settings.environment == "development"

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload
    )


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="ignitio-ai-tutor - LangGraph-based AI learning assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run CLI mode with sample query
    uv run python main.py

    # Start API server
    uv run python main.py --api

    # Start API server with custom port
    uv run python main.py --api --port 8080

    # Start API server with auto-reload (development)
    uv run python main.py --api --reload
"""
    )

    parser.add_argument(
        "--api",
        action="store_true",
        help="Run in API server mode instead of CLI mode"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind API server to (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server (default: 8000)"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (API mode only)"
    )

    args = parser.parse_args()

    if args.api:
        run_api(host=args.host, port=args.port, reload=args.reload)
    else:
        run_cli()


if __name__ == "__main__":
    main()