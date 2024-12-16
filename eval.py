from langsmith import traceable, aevaluate
from request import generate_prompt
from prompts import EVALUATION_PROMPT, SYSTEM_PROMPT
from clients import model_kwargs, openai_client, smol_model_kwargs
import asyncio

# The name or UUID of the LangSmith dataset to evaluate on
data = "rag_evaluation_dataset"

# A string to prefix the experiment name with
experiment_prefix = "RAG Test Dataset Evaluation"

@traceable
async def agent(inputs: dict) -> dict:
    """RAG agent that processes questions"""
    question = inputs["messages"][-1]["content"]
    
    prompt = await generate_prompt(question, inputs["messages"])

    # 5. Generate answer
    response = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        **model_kwargs
    )

    print("Agent Response: ", response)
    answer = response.choices[0].message.content
    
    return {
        "message": {
            "role": "assistant",
            "content": answer
        }
    }

async def score_evaluator(run, example) -> dict:
    """Evaluates the relevance and accuracy of the RAG system's answer"""
    question = run.inputs["inputs"]["messages"][-1]["content"]
    generated_answer = run.outputs["message"]["content"]
    reference_answer = example.outputs["message"]["content"]
    
    evaluation_prompt = f"""
    Question: {question}
    
    Generated Answer: {generated_answer}
    
    Reference Answer: {reference_answer}
    
    Score the generated answer from 0-5:
    5 = Correct and complete answer, may provide more information than the reference answer
    4 = Very good, minor differences from reference
    3 = Acceptable, but missing some details or slightly inaccurate
    2 = Partially correct but significant omissions or inaccuracies
    1 = Mostly incorrect or irrelevant
    0 = Completely wrong or unrelated
    
    Return only the number (0-5).
    """
    
    response = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": EVALUATION_PROMPT},
            {"role": "user", "content": evaluation_prompt}
        ],
        **smol_model_kwargs
    )
    
    print("Score Response: ", response)
    try:
        score = int(response.choices[0].message.content.strip())
        return {
            "key": "score",
            "score": score / 5,  # Normalize to 0-1
            "explanation": f"Answer score: {score}/5"
        }
    except ValueError:
        return {
            "key": "score",
            "score": 0,
            "explanation": "Failed to parse score"
        }

# Evaluate the target task
results = asyncio.run(aevaluate(
    agent,
    data=data,
    evaluators=[score_evaluator],
    experiment_prefix=experiment_prefix
))
