"""Test script for the builder node.

Tests:
1. LLM meme content generation
2. imgflip API meme creation
3. Concept map generation
4. Full builder_node execution
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from graph.state import graph_state
from nodes.builder.node import (
    builder_node,
    _generate_meme_content,
    _call_imgflip_api,
    _generate_concept_map,
    _get_fallback_concepts,
    POPULAR_TEMPLATES
)


def test_meme_content_generation():
    """Test that LLM generates valid meme content."""
    print("\n" + "="*60)
    print("TEST 1: Meme Content Generation")
    print("="*60)

    user_query = "Create a meme about debugging code at 3am"

    try:
        result = _generate_meme_content(user_query)
        print(f"User Query: {user_query}")
        print(f"Result: {json.dumps(result, indent=2)}")

        # Validate response structure
        assert "template_id" in result, "Missing template_id"
        assert "template_name" in result, "Missing template_name"
        assert "text_top" in result, "Missing text_top"
        assert "text_bottom" in result, "Missing text_bottom"

        print("\n[PASS] Meme content generated successfully")
        return result
    except Exception as e:
        print(f"\n[FAIL] {str(e)}")
        raise


def test_imgflip_api(template_id: str, text_top: str, text_bottom: str):
    """Test imgflip API meme generation."""
    print("\n" + "="*60)
    print("TEST 2: imgflip API Meme Creation")
    print("="*60)

    try:
        result = _call_imgflip_api(
            template_id=template_id,
            text_top=text_top,
            text_bottom=text_bottom
        )
        print(f"Template ID: {template_id}")
        print(f"Text Top: {text_top}")
        print(f"Text Bottom: {text_bottom}")
        print(f"Result: {json.dumps(result, indent=2)}")

        assert "url" in result, "Missing url in response"
        assert "page_url" in result, "Missing page_url in response"

        print(f"\n[PASS] Meme created successfully!")
        print(f"Meme URL: {result['url']}")
        return result
    except Exception as e:
        print(f"\n[FAIL] {str(e)}")
        raise


def test_concept_map_generation(user_query: str, meme_url: str):
    """Test LangChain concept map generation."""
    print("\n" + "="*60)
    print("TEST 3: Concept Map Generation")
    print("="*60)

    try:
        result = _generate_concept_map(user_query, {"url": meme_url})
        print(f"User Query: {user_query}")
        print(f"Meme URL: {meme_url}")
        print(f"Result: {json.dumps(result, indent=2)}")

        # Validate we have 3 concepts
        assert "concept_1" in result, "Missing concept_1"
        assert "concept_2" in result, "Missing concept_2"
        assert "concept_3" in result, "Missing concept_3"

        # Validate each concept has required fields
        for key in ["concept_1", "concept_2", "concept_3"]:
            concept = result[key]
            assert "name" in concept, f"Missing name in {key}"
            assert "description" in concept, f"Missing description in {key}"

        print("\n[PASS] Concept map generated successfully")
        return result
    except Exception as e:
        print(f"\n[FAIL] {str(e)}")
        raise


def test_full_builder_node():
    """Test the complete builder_node execution."""
    print("\n" + "="*60)
    print("TEST 4: Full Builder Node Execution")
    print("="*60)

    user_query = "Make a meme about Python indentation errors"

    # Create a test state
    state = graph_state(
        user_query=user_query,
        sub_tasks={}
    )

    try:
        result = builder_node(state)
        print(f"User Query: {user_query}")
        print(f"Result: {json.dumps(result, indent=2)}")

        # Validate all required outputs
        assert "meme_url" in result, "Missing meme_url"
        assert "meme_text" in result, "Missing meme_text"
        assert "concept_map" in result, "Missing concept_map"

        # Validate concept map structure
        assert len(result["concept_map"]) == 3, "Should have exactly 3 concepts"

        print("\n" + "-"*40)
        print("SUMMARY:")
        print(f"  Meme URL: {result['meme_url']}")
        print(f"  Meme Text: {result['meme_text']}")
        print(f"  Concepts: {list(result['concept_map'].keys())}")
        print("-"*40)

        print("\n[PASS] Builder node executed successfully")
        return result
    except Exception as e:
        print(f"\n[FAIL] {str(e)}")
        raise


def test_fallback_concepts():
    """Test fallback concepts function."""
    print("\n" + "="*60)
    print("TEST 5: Fallback Concepts")
    print("="*60)

    result = _get_fallback_concepts()
    print(f"Fallback Concepts: {json.dumps(result, indent=2)}")

    assert len(result) == 3, "Should have exactly 3 fallback concepts"
    print("\n[PASS] Fallback concepts available")


def test_rate_limiter():
    """Test the rate limiter functionality."""
    print("\n" + "="*60)
    print("TEST 6: Rate Limiter")
    print("="*60)

    from nodes.builder.node import _rate_limiter

    print(f"Max Requests: {_rate_limiter.max_requests}")
    print(f"Window Seconds: {_rate_limiter.window_seconds}")
    print(f"Current Request Times: {_rate_limiter.request_times}")

    assert _rate_limiter.max_requests == 5, "Rate limit should be 5 requests"
    assert _rate_limiter.window_seconds == 60, "Window should be 60 seconds"

    print("\n[PASS] Rate limiter configured correctly")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# BUILDER NODE TEST SUITE")
    print("#"*60)

    try:
        # Test 1: Meme content generation
        meme_content = test_meme_content_generation()

        # Test 2: imgflip API
        meme_result = test_imgflip_api(
            template_id=meme_content["template_id"],
            text_top=meme_content["text_top"],
            text_bottom=meme_content["text_bottom"]
        )

        # Test 3: Concept map generation
        test_concept_map_generation(
            user_query="Create a meme about debugging code at 3am",
            meme_url=meme_result["url"]
        )

        # Test 4: Full builder node
        test_full_builder_node()

        # Test 5: Fallback concepts
        test_fallback_concepts()

        # Test 6: Rate limiter
        test_rate_limiter()

        print("\n" + "#"*60)
        print("# ALL TESTS PASSED")
        print("#"*60 + "\n")

    except Exception as e:
        print("\n" + "#"*60)
        print(f"# TESTS FAILED: {str(e)}")
        print("#"*60 + "\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)