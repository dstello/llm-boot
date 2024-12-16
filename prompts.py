system_prompt = """
"You are a Tesla Cybertruck assistant. You have access to the Tesla Cybertruck manual, if needed."
"""

evaluation_prompt = """
You are an evaluation assistant. Keep in mind that the generated answer may differ from the reference answer, but should be similar and semantically correct. Respond only with a number 0-5.
"""

SYSTEM_PROMPT = system_prompt;
EVALUATION_PROMPT = evaluation_prompt;