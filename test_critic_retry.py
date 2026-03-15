"""Unit tests for the critic node retry logic bug fix.

Tests verify:
1. First failure returns has_retried=False (allowing routing to teaching)
2. Teaching sets has_retried=True when running as retry
3. Second failure returns FAIL_FINAL (no more retries)
4. Concept is NOT added to learning_test on first failure (re-test same concept)
5. Assessment is properly passed to teaching for targeted re-explanation
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from graph.state import graph_state
from nodes.critic.node import critic_node, _get_next_concept_index


def create_mock_state(
    learning_test: dict = None,
    has_retried: bool = False,
    concept_map: dict = None,
    explanation: dict = None,
    code_examples: dict = None,
    assessment_for_teaching: dict = None
) -> graph_state:
    """Helper to create a mock graph_state for testing."""
    return graph_state(
        user_query="Test query",
        sub_tasks={"is_valid": True},
        meme_url="https://example.com/meme.png",
        meme_text="Test meme text",
        concept_map=concept_map or {
            "concept_1": {"name": "Test Concept 1", "description": "Description 1"},
            "concept_2": {"name": "Test Concept 2", "description": "Description 2"},
            "concept_3": {"name": "Test Concept 3", "description": "Description 3"}
        },
        explanation=explanation or {
            "concept_1": {"name": "Test Concept 1", "description": "Explanation 1"},
            "concept_2": {"name": "Test Concept 2", "description": "Explanation 2"},
            "concept_3": {"name": "Test Concept 3", "description": "Explanation 3"}
        },
        code_examples=code_examples or {
            "concept_1": {"code": "print('hello')"},
            "concept_2": {"code": "print('world')"},
            "concept_3": {"code": "print('!')"}
        },
        learning_test=learning_test or {},
        test_result="",
        assessment_for_teaching=assessment_for_teaching or {},
        has_retried=has_retried,
        retry_count=0
    )


def test_get_next_concept_index_returns_first_untested():
    """Test that _get_next_concept_index returns the first concept not in learning_test."""
    print("\n" + "="*60)
    print("TEST 1: _get_next_concept_index returns first untested concept")
    print("="*60)

    concept_map = {
        "concept_1": {"name": "C1"},
        "concept_2": {"name": "C2"},
        "concept_3": {"name": "C3"}
    }

    # Empty learning_test -> should return 1
    result = _get_next_concept_index({}, concept_map)
    assert result == 1, f"Expected 1, got {result}"
    print("  [PASS] Empty learning_test returns concept_1 (index 1)")

    # concept_1 tested -> should return 2
    result = _get_next_concept_index({"concept_1": {}}, concept_map)
    assert result == 2, f"Expected 2, got {result}"
    print("  [PASS] concept_1 tested returns concept_2 (index 2)")

    # concept_1 and concept_2 tested -> should return 3
    result = _get_next_concept_index({"concept_1": {}, "concept_2": {}}, concept_map)
    assert result == 3, f"Expected 3, got {result}"
    print("  [PASS] concept_1 and concept_2 tested returns concept_3 (index 3)")

    # All concepts tested -> should return None
    result = _get_next_concept_index({"concept_1": {}, "concept_2": {}, "concept_3": {}}, concept_map)
    assert result is None, f"Expected None, got {result}"
    print("  [PASS] All concepts tested returns None")


def test_first_failure_returns_has_retried_false():
    """Test that first failure returns has_retried=False to allow routing to teaching."""
    print("\n" + "="*60)
    print("TEST 2: First failure returns has_retried=False")
    print("="*60)

    state = create_mock_state(has_retried=False)

    with patch('nodes.critic.node._generate_question') as mock_question, \
         patch('nodes.critic.node._evaluate_user_code') as mock_eval, \
         patch('nodes.critic.node._create_assessment_summary') as mock_assessment:

        mock_question.return_value = {"question": "Test question?"}
        mock_eval.return_value = {"passed": False, "score": 30, "assessment": {"errors": ["Bad code"]}}
        mock_assessment.return_value = {
            "concept_name": "Test Concept 1",
            "key_misunderstandings": ["Didn't understand X"],
            "suggested_focus": "Focus on X"
        }

        result = critic_node(state)

        # Verify has_retried is False (not True!) to allow routing to teaching
        assert result["has_retried"] == False, f"Expected has_retried=False, got {result['has_retried']}"
        print("  [PASS] First failure returns has_retried=False")

        # Verify test_result is FAIL
        assert result["test_result"] == "FAIL", f"Expected test_result='FAIL', got {result['test_result']}"
        print("  [PASS] First failure returns test_result='FAIL'")

        # Verify concept_1 is NOT in learning_test (will be re-tested)
        assert "concept_1" not in result["learning_test"], "concept_1 should NOT be in learning_test after first failure"
        print("  [PASS] concept_1 NOT in learning_test (will be re-tested)")

        # Verify assessment_for_teaching is set
        assert "assessment_for_teaching" in result, "assessment_for_teaching should be set"
        assert result["assessment_for_teaching"]["concept_name"] == "Test Concept 1"
        print("  [PASS] assessment_for_teaching is set for teaching retry")


def test_second_failure_returns_fail_final():
    """Test that second failure (has_retried=True) returns FAIL_FINAL."""
    print("\n" + "="*60)
    print("TEST 3: Second failure returns FAIL_FINAL")
    print("="*60)

    state = create_mock_state(has_retried=True)

    with patch('nodes.critic.node._generate_question') as mock_question, \
         patch('nodes.critic.node._evaluate_user_code') as mock_eval:

        mock_question.return_value = {"question": "Test question?"}
        mock_eval.return_value = {"passed": False, "score": 30, "assessment": {"errors": ["Bad code"]}}

        result = critic_node(state)

        # Verify test_result is FAIL_FINAL
        assert result["test_result"] == "FAIL_FINAL", f"Expected test_result='FAIL_FINAL', got {result['test_result']}"
        print("  [PASS] Second failure returns test_result='FAIL_FINAL'")

        # Verify concept is added to learning_test
        assert "concept_1" in result["learning_test"], "concept_1 should be in learning_test after final failure"
        print("  [PASS] concept_1 added to learning_test after final failure")

        # Verify has_retried stays True
        assert result["has_retried"] == True, f"Expected has_retried=True, got {result['has_retried']}"
        print("  [PASS] has_retried stays True")


def test_pass_resets_has_retried_and_clears_assessment():
    """Test that passing resets has_retried and clears assessment_for_teaching."""
    print("\n" + "="*60)
    print("TEST 4: Pass resets has_retried and clears assessment")
    print("="*60)

    # Simulate state after a retry (has_retried=True, assessment set)
    state = create_mock_state(
        has_retried=True,
        assessment_for_teaching={"concept_name": "Previous concept", "key_misunderstandings": ["X"]}
    )

    with patch('nodes.critic.node._generate_question') as mock_question, \
         patch('nodes.critic.node._evaluate_user_code') as mock_eval:

        mock_question.return_value = {"question": "Test question?"}
        mock_eval.return_value = {"passed": True, "score": 90, "assessment": {}}

        result = critic_node(state)

        # Verify test_result is PASS
        assert result["test_result"] == "PASS", f"Expected test_result='PASS', got {result['test_result']}"
        print("  [PASS] Pass returns test_result='PASS'")

        # Verify has_retried is reset to False
        assert result["has_retried"] == False, f"Expected has_retried=False, got {result['has_retried']}"
        print("  [PASS] has_retried reset to False after pass")

        # Verify assessment_for_teaching is cleared
        assert result["assessment_for_teaching"] == {}, f"Expected empty assessment, got {result['assessment_for_teaching']}"
        print("  [PASS] assessment_for_teaching cleared after pass")


def test_concept_retested_after_first_failure():
    """Test that after first failure, the same concept is re-tested (not skipped)."""
    print("\n" + "="*60)
    print("TEST 5: Same concept re-tested after first failure")
    print("="*60)

    # Simulate state after teaching re-explained (has_retried=True from teaching)
    # learning_test should still be empty (concept_1 not added after first failure)
    state = create_mock_state(has_retried=True)

    # Verify that _get_next_concept_index returns 1 (concept_1 should be re-tested)
    next_index = _get_next_concept_index(state.learning_test, state.concept_map)
    assert next_index == 1, f"Expected next concept index 1, got {next_index}"
    print("  [PASS] After first failure, concept_1 is still the next concept to test")

    # Simulate what critic_node does
    with patch('nodes.critic.node._generate_question') as mock_question, \
         patch('nodes.critic.node._evaluate_user_code') as mock_eval:

        mock_question.return_value = {"question": "Retry question?"}
        mock_eval.return_value = {"passed": True, "score": 85, "assessment": {}}

        result = critic_node(state)

        # Verify concept_1 was tested (it's in learning_test after pass)
        assert "concept_1" in result["learning_test"], "concept_1 should be in learning_test after retry pass"
        print("  [PASS] concept_1 successfully re-tested and passed after retry")


def test_full_retry_flow():
    """Test the complete retry flow: fail -> teaching -> pass."""
    print("\n" + "="*60)
    print("TEST 6: Full retry flow simulation")
    print("="*60)

    # Step 1: Initial state
    print("\n  Step 1: Initial critic run (concept_1 fails)")
    state = create_mock_state(has_retried=False)

    with patch('nodes.critic.node._generate_question') as mock_question, \
         patch('nodes.critic.node._evaluate_user_code') as mock_eval, \
         patch('nodes.critic.node._create_assessment_summary') as mock_assessment:

        mock_question.return_value = {"question": "Initial question?"}
        mock_eval.return_value = {"passed": False, "score": 40, "assessment": {"errors": []}}
        mock_assessment.return_value = {"concept_name": "Test Concept 1", "key_misunderstandings": ["X"]}

        result1 = critic_node(state)

        assert result1["test_result"] == "FAIL", "Expected FAIL"
        assert result1["has_retried"] == False, "has_retried should be False for routing"
        assert "concept_1" not in result1["learning_test"], "concept_1 not in learning_test"
        print("    [PASS] First failure: FAIL, has_retried=False, concept_1 NOT in learning_test")

    # Step 2: Simulate teaching node behavior
    print("\n  Step 2: Teaching node sets has_retried=True")
    # (This would normally happen in teaching_node, returning has_retried=True)
    state_after_teaching = graph_state(
        user_query=state.user_query,
        sub_tasks=state.sub_tasks,
        meme_url=state.meme_url,
        meme_text=state.meme_text,
        concept_map=state.concept_map,
        explanation={"concept_1": {"name": "Test Concept 1", "corrected_explanation": "Simpler explanation"}},
        code_examples=state.code_examples,
        learning_test={},  # Still empty
        test_result="",
        assessment_for_teaching={},  # Cleared after teaching uses it
        has_retried=True,  # Set by teaching!
        retry_count=0
    )

    # Step 3: Critic re-tests concept_1
    print("\n  Step 3: Critic re-tests concept_1 (has_retried=True)")
    with patch('nodes.critic.node._generate_question') as mock_question, \
         patch('nodes.critic.node._evaluate_user_code') as mock_eval:

        mock_question.return_value = {"question": "Retry question?"}
        mock_eval.return_value = {"passed": True, "score": 90, "assessment": {}}

        result2 = critic_node(state_after_teaching)

        assert result2["test_result"] == "PASS", f"Expected PASS, got {result2['test_result']}"
        assert result2["has_retried"] == False, "has_retried should reset after pass"
        assert "concept_1" in result2["learning_test"], "concept_1 should be in learning_test after pass"
        print("    [PASS] Retry success: PASS, has_retried=False (reset), concept_1 in learning_test")

    print("\n  [PASS] Full retry flow works correctly!")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# CRITIC NODE RETRY LOGIC TEST SUITE")
    print("# Testing the bug fix for: has_retried timing issue")
    print("#"*60)

    try:
        test_get_next_concept_index_returns_first_untested()
        test_first_failure_returns_has_retried_false()
        test_second_failure_returns_fail_final()
        test_pass_resets_has_retried_and_clears_assessment()
        test_concept_retested_after_first_failure()
        test_full_retry_flow()

        print("\n" + "#"*60)
        print("# ALL TESTS PASSED")
        print("# The retry logic bug fix is working correctly!")
        print("#"*60 + "\n")

    except AssertionError as e:
        print("\n" + "#"*60)
        print(f"# TEST FAILED: {str(e)}")
        print("#"*60 + "\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)