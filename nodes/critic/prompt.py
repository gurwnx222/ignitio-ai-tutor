"""Prompt templates for the critic node.

The critic agent:
1. Generates a coding question for each concept taught
2. Evaluates user's code answer
3. Provides detailed feedback on incorrect answers
4. Tracks retry count to enforce max 1 retry loop
"""


QUESTION_GENERATION_PROMPT = """You are a coding assessment expert creating a practical coding test.

The user has just learned about this LangChain concept:
Concept Name: {concept_name}
Concept Description: {concept_description}
Code Example they saw:
{code_example}

## Your Task

Create ONE practical coding question that tests the user's understanding of this concept.

The question should:
1. Be answerable by writing 5-15 lines of code
2. Test the core understanding of the concept (not just syntax)
3. Be directly related to the code example they studied
4. Have a clear, unambiguous correct answer

## Example Question Format

If the concept is "LCEL (LangChain Expression Language)" and the example shows chain = prompt | model | output_parser, the question could be:

"How would you create a chain that connects a ChatPromptTemplate, a ChatOpenAI model, and a StrOutputParser? Write the code to create this chain."

## Important

- The question must require the user to WRITE CODE (not just explain)
- The question should be solvable using the code example provided
- Avoid trick questions or edge cases
- Focus on practical application of the concept

Respond with ONLY a JSON object (no markdown formatting):
{{
    "question": "Your coding question here",
    "concept_tested": "{concept_name}",
    "difficulty": "beginner|intermediate",
    "expected_keywords": ["keyword1", "keyword2", "keyword3"]
}}
"""


CODE_EVALUATION_PROMPT = """You are a patient code reviewer evaluating a beginner's code answer.

## Concept Being Tested
Concept Name: {concept_name}
Concept Description: {concept_description}

## The Question
{question}

## The Code Example They Studied
{code_example}

## The User's Code Answer
{user_code}

## Your Task

Evaluate the user's code answer. Be fair and constructive.

Check for:
1. **Correctness** - Does the code solve the problem correctly?
2. **Concept Understanding** - Does it demonstrate understanding of the concept?
3. **Syntax** - Is the syntax correct (or close enough for a beginner)?
4. **Completeness** - Is the code complete enough to work?

## Evaluation Criteria

- A beginner who understands the concept may make minor syntax errors
- The code doesn't need to be perfect, just demonstrate understanding
- Partial credit if they got the main idea but missed some details
- Be encouraging even when pointing out errors

Respond with ONLY a JSON object (no markdown formatting):
{{
    "passed": true|false,
    "score": 0-100,
    "assessment": {{
        "strengths": ["what they did well"],
        "errors": ["specific mistakes made"],
        "missing_concepts": ["concepts they didn't understand or apply correctly"],
        "suggestions": ["how they can improve"]
    }},
    "correct_answer": "the correct code answer (for comparison)",
    "explanation_for_retry": "If they failed, explain in simpler terms what they got wrong and how to fix it. This will be used for the teaching retry."
}}
"""


ASSESSMENT_SUMMARY_PROMPT = """You are creating a summary of a user's learning test results for the teaching agent.

## Test Results
- Concept: {concept_name}
- Attempt: {attempt_number} of 2
- Passed: {passed}
- Score: {score}

## Detailed Assessment
{detailed_assessment}

## Your Task

Create a concise summary for the teaching agent that will help them adjust their re-explanation if this was a failure.

If the user passed, celebrate briefly and move on.
If the user failed, identify the key misunderstandings that need simpler explanation.

Respond with ONLY a JSON object (no markdown formatting):
{{
    "summary": "Brief summary of results",
    "key_misunderstandings": ["what the user didn't understand"],
    "suggested_focus": "What the teaching agent should focus on in the retry"
}}
"""