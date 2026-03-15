"""Session management for the ignitio-ai-tutor pipeline.

Since the graph has stateful flow (user answers test questions separately),
we need to manage session state between requests.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass, field
from threading import Lock

from graph.state import graph_state


@dataclass
class Session:
    """Represents a user session with the tutor pipeline."""

    id: str
    state: graph_state
    created_at: datetime
    expires_at: datetime
    current_step: str = "init"  # init, teaching, test_ready, completed
    test_questions: Optional[Dict] = None

    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict:
        """Convert session to dictionary for serialization."""
        return {
            "id": self.id,
            "state": self.state.model_dump(),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "current_step": self.current_step,
            "test_questions": self.test_questions
        }


class SessionManager:
    """Manages user sessions for the tutor pipeline.

    Uses in-memory storage with UUID-based session IDs.
    Thread-safe for concurrent access.
    """

    def __init__(self, max_sessions: int = 1000, session_ttl_minutes: int = 60):
        """
        Initialize the session manager.

        Args:
            max_sessions: Maximum number of concurrent sessions
            session_ttl_minutes: Session time-to-live in minutes
        """
        self._sessions: Dict[str, Session] = {}
        self._lock = Lock()
        self._max_sessions = max_sessions
        self._session_ttl = timedelta(minutes=session_ttl_minutes)

    def create_session(self, initial_state: graph_state) -> Session:
        """
        Create a new session with an initial state.

        Args:
            initial_state: The initial graph state for the session

        Returns:
            Session: The created session

        Raises:
            RuntimeError: If max sessions limit is reached
        """
        with self._lock:
            # Check if we've hit the session limit
            if len(self._sessions) >= self._max_sessions:
                self._cleanup_expired_sessions()
                if len(self._sessions) >= self._max_sessions:
                    raise RuntimeError("Maximum session limit reached")

            session_id = str(uuid.uuid4())
            now = datetime.now()

            session = Session(
                id=session_id,
                state=initial_state,
                created_at=now,
                expires_at=now + self._session_ttl,
                current_step="init"
            )

            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieve a session by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            Session or None if not found or expired
        """
        with self._lock:
            session = self._sessions.get(session_id)

            if session is None:
                return None

            if session.is_expired():
                del self._sessions[session_id]
                return None

            return session

    def update_session(self, session_id: str, state: graph_state, current_step: Optional[str] = None) -> Session:
        """
        Update a session's state and optionally its step.

        Args:
            session_id: The session ID to update
            state: The new graph state
            current_step: Optional new step

        Returns:
            Session: The updated session

        Raises:
            KeyError: If session not found
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")

            session = self._sessions[session_id]
            session.state = state

            if current_step is not None:
                session.current_step = current_step

            # Extend expiration time on activity
            session.expires_at = datetime.now() + self._session_ttl

            return session

    def set_test_questions(self, session_id: str, questions: Dict) -> None:
        """
        Store test questions for a session.

        Args:
            session_id: The session ID
            questions: The test questions dictionary

        Raises:
            KeyError: If session not found
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")

            self._sessions[session_id].test_questions = questions

    def get_test_questions(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve test questions for a session.

        Args:
            session_id: The session ID

        Returns:
            Dict or None if not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return session.test_questions

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session by ID.

        Args:
            session_id: The session ID to delete

        Returns:
            bool: True if deleted, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def _cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions from the store.

        Returns:
            int: Number of sessions removed
        """
        now = datetime.now()
        expired_ids = [
            sid for sid, session in self._sessions.items()
            if session.is_expired()
        ]

        for sid in expired_ids:
            del self._sessions[sid]

        return len(expired_ids)

    def get_session_count(self) -> int:
        """Get the current number of active sessions."""
        with self._lock:
            return len(self._sessions)


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """
    Get or create the global session manager instance.

    Returns:
        SessionManager: The global session manager
    """
    global _session_manager
    if _session_manager is None:
        from api.config import settings
        _session_manager = SessionManager(
            max_sessions=settings.max_sessions,
            session_ttl_minutes=settings.session_max_age_minutes
        )
    return _session_manager