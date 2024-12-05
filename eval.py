from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable, evaluate

from dotenv import load_dotenv
load_dotenv()

client = wrap_openai(OpenAI())

from prompts import TEST_PROMPT_1, TEST_PROMPT_2

@traceable
def agent(inputs: dict) -> dict:
    messages = [
        {"role": "system", "content": TEST_PROMPT_2},
        *inputs["messages"]
    ]

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2
    )

    return {
        "message": {
            "role": "assistant",
            "content": result.choices[0].message.content
        }
    }

# The name or UUID of the LangSmith dataset to evaluate on.
data = "llm-bootcamp-finance-questions"

# A string to prefix the experiment name with.
experiment_prefix = "finance-questions-dataset"

def correctness_evaluator(run, example) -> dict:
    """
    Evaluates the correctness of the responses given by the agent
    
    Args:
        run: Contains the run information including inputs and outputs
        example: Contains the reference example if available
    
    Returns:
        Dictionary with score (0-1) and explanation
    """
    # Extract the original LeetCode problem from inputs
    question = run.inputs["inputs"]["messages"][-1]["content"]
    
    # Extract the model's generated responses
    response = run.outputs["message"]["content"]
    
    # Rest of the evaluation logic remains the same
    evaluation_prompt = f"""
    Given this question:
    {question}

    Evaluate the response given by the agent:
    {response}
    
    Score from 0-4:
    4 = The response is returned in the correct format and is correct
    3 = The response is returned in the correct format but is incorrect
    2 = The response is partially correct, incomplete or has some issues
    1 = The response is incorrect
    0 = The response is completely incorrect, and may not even be related to the question
    
    Return only the number (0-4).
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant to audit the correctness of a response given by an agent. Your job is to ensure the response is correct by evaluating it against the question.Respond only with a number 0-4."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0
    )
    
    try:
        score = int(response.choices[0].message.content.strip())
        return {
            "key": "correctness score",
            "score": score / 4,  # Normalize to 0-1
            "explanation": f"Response correctness score: {score}/4"
        }
    except ValueError:
        return {
            "key": "correctness score",
            "score": 0,
            "explanation": "Failed to parse score"
        }

# List of evaluators to score the outputs of target task
evaluators = [
    correctness_evaluator
]

# Evaluate the target task
results = evaluate(
    agent,
    data=data,
    evaluators=evaluators,
    experiment_prefix=experiment_prefix
)