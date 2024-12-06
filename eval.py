from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable, evaluate

from dotenv import load_dotenv
load_dotenv()

client = wrap_openai(OpenAI())

from prompts import SYSTEM_PROMPT

@traceable
def agent(inputs: dict) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *inputs["messages"]
    ]

    result = client.chat.completions.create(
        model="gpt-4o-turbo",
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

Accuracy is the most important metric. The numbers in the answer should be correct, and the answer should be exhaustive based on the given data. Give a score based on the following:
Accuracy Score (0-4):
    4: The response is completely accurate and based on reliable data.
    3: The response is mostly accurate with minor inaccuracies that do not significantly affect the overall correctness.
    2: The response is partially accurate but contains some inaccuracies that could mislead the user.
    1: The response is largely inaccurate and could lead to incorrect conclusions.
    0: The response is completely inaccurate or unrelated to the question.

Given the answer should be concise, and, if applicable, formatted in the as the user has asked, give a score based on the following:
Formatting Score (0-4):
    4: The response is well-formatted, easy to read, and follows a logical structure.
    3: The response is generally well-formatted but has minor issues in structure or readability.
    2: The response is somewhat disorganized, affecting readability and comprehension.
    1: The response is poorly formatted, making it difficult to understand.
    0: The response lacks any coherent formatting, rendering it unreadable.

The answer should be clear and concise. Give a score based on the following:
Clarity Score (0-4):
    4: The response is clear, concise, and easy to understand, with no ambiguity.
    3: The response is mostly clear but contains some ambiguous or unclear elements.
    2: The response is somewhat unclear or contains jargon that may confuse the user.
    1: The response is unclear and difficult to understand.
    0: The response is completely unclear and incomprehensible.

If the question asks for advice, guidance, or goals, give a score based on the following:
Insight Score (0-4):
    4: The response provides valuable insights and actionable advice tailored to the user's transaction data and needs.
    3: The response offers useful insights but lacks depth or specific action steps.
    2: The response provides limited insights that may not be very useful.
    1: The response offers little to no insight or actionable advice.
    0: The question does not ask for advice, guidance, or goals.
    
    Return only the numbers (0-4) in a comma separated list: accuracy, formatting, clarity, insight.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant to audit the correctness of a response given by the agent. Your job is to ensure the response is correct by evaluating it against the question. Respond only with numbers 0-4 in a comma separated list: accuracy, formatting, clarity, insight."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0
    )
    
    try:
        score = response.choices[0].message.content.strip().split(',')
        score = [int(s) for s in score]
        return [{
            "key": "correctness score",
            "score": score[0] / 4,  # Normalize to 0-1
            "explanation": f"Response correctness score: {score[0]}/4"
        },
        {
            "key": "formatting score",
            "score": score[1] / 4,  # Normalize to 0-1
            "explanation": f"Response formatting score: {score[1]}/4"
        },
        {
            "key": "clarity score",
            "score": score[2] / 4,  # Normalize to 0-1
            "explanation": f"Response clarity score: {score[2]}/4"
        },
        {
            "key": "insight score",
            "score": score[3] / 4,  # Normalize to 0-1
            "explanation": f"Response insight score: {score[3]}/4"
        }]
    except ValueError:
        return [{
            "key": "correctness score",
            "score": 0,
            "explanation": "Failed to parse score"
        },
        {
            "key": "formatting score",
            "score": 0,
            "explanation": "Failed to parse score"
        },
        {
            "key": "clarity score",
            "score": 0,
            "explanation": "Failed to parse score"
        },
        {
            "key": "insight score",
            "score": 0,
            "explanation": "Failed to parse score"
        }]

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