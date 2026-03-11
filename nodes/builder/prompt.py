"""Prompt templates for the builder node."""


MEME_GENERATION_PROMPT = """You are a meme creation expert. Your task is to select the best meme template and create funny, relevant caption text for the user's request.

Available meme templates: {templates}

User's meme request: {user_query}

Guidelines for great memes:
1. Choose a template that fits the topic/emotion naturally
2. Top text should set up the joke/situation
3. Bottom text should deliver the punchline or twist
4. Keep text short and punchy (under 50 chars each line)
5. Make it relatable and funny

Respond with ONLY a JSON object:
{{
    "template": "template_name from the available templates",
    "text_top": "Top line of the meme",
    "text_bottom": "Bottom line of the meme",
    "reasoning": "Brief explanation of why this template and text work"
}}

Do not include any other text or markdown formatting."""


CONCEPT_MAP_PROMPT = """You are a LangChain expert. Identify exactly 3 core LangChain concepts that would be used to build an AI-powered meme generator like this one.

User's meme request: {user_query}
Generated meme URL: {meme_url}

The concepts should be foundational LangChain building blocks that enable:
1. Understanding the user's meme request
2. Generating appropriate meme text
3. Creating the workflow to produce memes

Respond with ONLY a JSON object:
{{
    "concept_1": {{
        "name": "LangChain Concept Name",
        "description": "Brief description of what this concept does",
        "relevance": "Why this concept is essential for building the meme generator"
    }},
    "concept_2": {{
        "name": "LangChain Concept Name",
        "description": "Brief description of what this concept does",
        "relevance": "Why this concept is essential for building the meme generator"
    }},
    "concept_3": {{
        "name": "LangChain Concept Name",
        "description": "Brief description of what this concept does",
        "relevance": "Why this concept is essential for building the meme generator"
    }}
}}

Do not include any other text or markdown formatting."""


# Template descriptions for better template selection
TEMPLATE_DESCRIPTIONS = """
Template Guide:
- drake: Comparing two things, one preferred over another
- distracted_boyfriend: Temptation, comparing priorities
- two_buttons: Difficult choice between two options
- change_my_mind: Controversial or bold statements
- expanding_brain: Progressively more absurd/brilliant ideas
- this_is_fine: Accepting a bad situation calmly
- surprised_pikachu: Unexpected consequences of actions
- one_does_not_simply: Difficult tasks that seem simple
- success_kid: Small victories or achievements
- bad_luck_brian: Unfortunate situations
- roll_safe: Obvious or hacky solutions
- monkey_puppet: Trying to look innocent while doing something wrong
- hide_the_pain_harold: Smiling through pain or awkwardness
- stonks: Bad financial decisions or misguided confidence
"""