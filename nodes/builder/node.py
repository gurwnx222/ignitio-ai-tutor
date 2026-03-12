"""Builder node for generating memes via imgflip API.

This node:
1. Uses LLM to select an appropriate meme template and generate caption text
2. Calls imgflip API to generate the meme image
3. Returns meme_url, meme_text, and concept_map to the graph state

Rate limit: 5 requests per minute (enforced via in-memory tracker)
"""

import time
import json
import requests
from typing import Optional
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage

from core.llm import get_llm
from graph.state import graph_state
from nodes.builder.prompt import MEME_GENERATION_PROMPT, CONCEPT_MAP_PROMPT


# ============================================================================
# Rate Limiter: 5 requests per minute
# ============================================================================
@dataclass
class RateLimiter:
    """Simple in-memory rate limiter for imgflip API calls.

    Limits to 5 requests per minute (12 second minimum between requests).
    """
    max_requests: int = 5
    window_seconds: int = 60
    request_times: list = field(default_factory=list)

    def wait_if_needed(self) -> None:
        """Block until a request can be made within rate limits."""
        current_time = time.time()

        # Remove requests older than the window
        self.request_times = [
            t for t in self.request_times
            if current_time - t < self.window_seconds
        ]

        # If at limit, wait until oldest request exits the window
        if len(self.request_times) >= self.max_requests:
            wait_time = self.window_seconds - (current_time - self.request_times[0])
            if wait_time > 0:
                time.sleep(wait_time)
            # Clean up again after waiting
            self.request_times = [
                t for t in self.request_times
                if time.time() - t < self.window_seconds
            ]

        # Record this request
        self.request_times.append(time.time())


# Global rate limiter instance
_rate_limiter = RateLimiter(max_requests=5, window_seconds=60)


import os
from dotenv import load_dotenv

load_dotenv()

# PSEUDO CODE: Load credentials from environment
IMGFLIP_USERNAME = os.getenv("IMGFLIP_USERNAME")
IMGFLIP_PASSWORD = os.getenv("IMGFLIP_PASSWORD")

# imgflip API endpoints
IMGFLIP_CAPTION_URL = "https://api.imgflip.com/caption_image"
IMGFLIP_TEMPLATES_URL = "https://api.imgflip.com/get_memes"

# Popular meme templates (fetched from imgflip API)
POPULAR_TEMPLATES = {
    "drake_hotline_bling": "181913649",
    "drake": "181913649",  # alias
    "distracted_boyfriend": "112126428",
    "two_buttons": "87743020",
    "change_my_mind": "129242436",
    "expanding_brain": "93895088",
    "this_is_fine": "55311130",
    "surprised_pikachu": "155067746",
    "one_does_not_simply": "61579",
    "success_kid": "61544",
    "roll_safe": "89370399",
    "monkey_puppet": "148909805",
    "hide_the_pain_harold": "27813981",
    "bad_luck_brian": "61539",
    "stonks": "155067746",  # using surprised_pikachu as fallback for stonks
}

# Builder Node

def builder_node(state: graph_state) -> dict:
    """
    Builder node that generates a meme via imgflip API.

    Process:
    1. Use LLM to select meme template and generate text
    2. Call imgflip API with rate limiting
    3. Generate concept_map (3 LangChain concepts)
    4. Return meme_url, meme_text, concept_map

    Args:
        state: Current graph state with user_query and sub_tasks

    Returns:
        dict: Updated state with meme_url, meme_text, concept_map
    """
    user_query = state.user_query
    sub_tasks = state.sub_tasks

    # Step 1: Use LLM to generate meme content
    try:
        meme_content = _generate_meme_content(user_query)
    except Exception as e:
        return _error_response(f"Failed to generate meme content: {str(e)}")

    # Step 2: Call imgflip API with rate limiting
    try:
        _rate_limiter.wait_if_needed()
        meme_result = _call_imgflip_api(
            template_id=meme_content["template_id"],
            text_top=meme_content.get("text_top", ""),
            text_bottom=meme_content.get("text_bottom", "")
        )
    except Exception as e:
        return _error_response(f"Failed to create meme via imgflip: {str(e)}")

    # Step 3: Generate concept map
    try:
        concept_map = _generate_concept_map(user_query, meme_result)
    except Exception as e:
        # Use fallback concepts if generation fails
        concept_map = _get_fallback_concepts()

    return {
        "meme_url": meme_result.get("url", ""),
        "meme_text": f"{meme_content.get('text_top', '')} | {meme_content.get('text_bottom', '')}".strip(" |"),
        "concept_map": concept_map
    }


def _generate_meme_content(user_query: str) -> dict:
    """
    Use LLM to select a meme template and generate caption text.

    Args:
        user_query: User's meme request

    Returns:
        dict: {template_id, template_name, text_top, text_bottom}
    """
    llm = get_llm()

    prompt = MEME_GENERATION_PROMPT.format(
        user_query=user_query,
        templates=json.dumps(list(POPULAR_TEMPLATES.keys()))
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    content = _parse_json_response(response.content)

    # Handle both template name and direct template_id from LLM response
    template_name = content.get("template", content.get("template_name", "drake"))

    # Check if LLM returned a template_id directly
    if "template_id" in content and content["template_id"]:
        template_id = content["template_id"]
    else:
        # Map template name to ID
        template_id = POPULAR_TEMPLATES.get(template_name, POPULAR_TEMPLATES["drake"])

    return {
        "template_id": template_id,
        "template_name": template_name,
        "text_top": content.get("text_top", ""),
        "text_bottom": content.get("text_bottom", "")
    }


def _call_imgflip_api(
    template_id: str,
    text_top: str = "",
    text_bottom: str = ""
) -> dict:
    """
    Call imgflip API to generate a meme image.

    Args:
        template_id: imgflip template ID
        text_top: Top caption text
        text_bottom: Bottom caption text

    Returns:
        dict: {url: str, page_url: str}

    Raises:
        Exception: If API call fails
    """
    # Validate credentials
    if not IMGFLIP_USERNAME or not IMGFLIP_PASSWORD:
        raise ValueError(
            "imgflip credentials not configured. "
            "Set IMGFLIP_USERNAME and IMGFLIP_PASSWORD in .env file. "
            "Get credentials from https://imgflip.com/signup"
        )

    payload = {
        "template_id": template_id,
        "username": IMGFLIP_USERNAME,
        "password": IMGFLIP_PASSWORD,
        "text0": text_top[:100] if text_top else "",  # imgflip has text limits
        "text1": text_bottom[:100] if text_bottom else "",
        "max_font_size": 35,
        "font": "Impact"
    }

    try:
        response = requests.post(
            IMGFLIP_CAPTION_URL,
            data=payload,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()

        if not data.get("success", False):
            error_msg = data.get("error_message", "Unknown imgflip API error")
            raise Exception(f"imgflip API error: {error_msg}")

        return {
            "url": data["data"]["url"],
            "page_url": data["data"]["page_url"]
        }

    except requests.exceptions.Timeout:
        raise Exception("imgflip API request timed out after 30 seconds")
    except requests.exceptions.RequestException as e:
        raise Exception(f"imgflip API request failed: {str(e)}")


def _generate_concept_map(user_query: str, meme_result: dict) -> dict:
    """
    Generate 3 LangChain concepts that underpin the meme generator.

    Args:
        user_query: User's meme request
        meme_result: Result from imgflip API

    Returns:
        dict: {concept_1: {...}, concept_2: {...}, concept_3: {...}}
    """
    llm = get_llm()

    prompt = CONCEPT_MAP_PROMPT.format(
        user_query=user_query,
        meme_url=meme_result.get("url", "")
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return _parse_json_response(response.content)


def _get_fallback_concepts() -> dict:
    """Return fallback concepts if generation fails."""
    return {
        "concept_1": {
            "name": "Language Models",
            "description": "AI models that understand and generate human-like text"
        },
        "concept_2": {
            "name": "Prompt Engineering",
            "description": "Crafting effective instructions to guide AI behavior"
        },
        "concept_3": {
            "name": "Chains",
            "description": "Sequencing multiple AI operations into a workflow"
        }
    }


def _parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response, handling formatting issues."""
    content = content.strip()

    # Remove markdown code blocks
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
        return {}


def _error_response(message: str) -> dict:
    """Create an error response for the graph state."""
    return {
        "meme_url": "",
        "meme_text": f"Error: {message}",
        "concept_map": _get_fallback_concepts()
    }


# Export for LangGraph
__all__ = ["builder_node"]