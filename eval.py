from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable, evaluate
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize clients
wcd_url = os.environ["WCD_URL"]
wcd_api_key = os.environ["WCD_API_KEY"]

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,
    auth_credentials=Auth.api_key(wcd_api_key),
    skip_init_checks=True
)

base_openai_client = OpenAI()
openai_client = wrap_openai(base_openai_client)

def get_answer(query: str, weaviate_client, openai_client) -> str:
    """Get an answer using RAG pipeline"""
    # 1. Generate embedding
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    query_embedding = response.data[0].embedding

    # 2. Retrieve chunks
    collection = weaviate_client.collections.get("TeslaCybertruck")
    similar_texts = collection.query.near_vector(
        near_vector=query_embedding,
        limit=3,
        return_properties=["text"],
        return_metadata=MetadataQuery(distance=True)
    )

    # 3. Combine contexts
    context_str = "\n\n---\n\n".join(
        [doc.properties["text"] for doc in similar_texts.objects]
    )
    
    # 4. Create prompt
    prompt = f"""Answer the question using ONLY the information provided in the context below. 
    Do not add any general knowledge or information not contained in the context.

    Context:
    {context_str}

    Question: {query}

    Answer:"""

    # 5. Generate answer
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

@traceable
def agent(inputs: dict) -> dict:
    """RAG agent that processes questions"""
    question = inputs["messages"][-1]["content"]
    
    answer = get_answer(
        query=question,
        weaviate_client=weaviate_client,
        openai_client=openai_client
    )
    
    return {
        "message": {
            "role": "assistant",
            "content": answer
        }
    }

def answer_relevance_evaluator(run, example) -> dict:
    """Evaluates the relevance and accuracy of the RAG system's answer"""
    question = run.inputs["inputs"]["messages"][-1]["content"]
    generated_answer = run.outputs["message"]["content"]
    reference_answer = example.outputs["message"]["content"]
    
    evaluation_prompt = f"""
    Question: {question}
    
    Generated Answer: {generated_answer}
    
    Reference Answer: {reference_answer}
    
    Score the generated answer from 0-5:
    5 = Perfect match with reference, complete and accurate
    4 = Very good, minor differences from reference
    3 = Acceptable, but missing some details or slightly inaccurate
    2 = Partially correct but significant omissions or inaccuracies
    1 = Mostly incorrect or irrelevant
    0 = Completely wrong or unrelated
    
    Return only the number (0-5).
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an evaluation assistant. Respond only with a number 0-5."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0
    )
    
    try:
        score = int(response.choices[0].message.content.strip())
        return {
            "key": "answer_relevance",
            "score": score / 5,  # Normalize to 0-1
            "explanation": f"Answer relevance score: {score}/5"
        }
    except ValueError:
        return {
            "key": "answer_relevance",
            "score": 0,
            "explanation": "Failed to parse score"
        }

# The name or UUID of the LangSmith dataset to evaluate on
data = "rag_evaluation_dataset"

# A string to prefix the experiment name with
experiment_prefix = "RAG Test Dataset Evaluation"

# List of evaluators
evaluators = [answer_relevance_evaluator]

# Evaluate the target task
results = evaluate(
    agent,
    data=data,
    evaluators=evaluators,
    experiment_prefix=experiment_prefix
)

# Clean up
weaviate_client.close()