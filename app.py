import chainlit as cl
import openai
import os
import base64
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from prompts import SYSTEM_PROMPT

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

from dotenv import load_dotenv

load_dotenv()

ENABLE_SYSTEM_PROMPT=True

# OPEN AI
api_key = os.getenv("OPENAI_API_KEY")
endpoint_url = "https://api.openai.com/v1"
model_kwargs = {
    "model": "gpt-4o-mini",
    "temperature": 1.2,
    "max_tokens": 500
}
smol_model_kwargs = {
    "model": "gpt-4o-mini",
    "temperature": 0,
    "max_tokens": 500
}

# ANTHROPIC
# api_key = os.getenv("ANTHROPIC_API_KEY")
# endpoint_url = "https://api.anthropic.com/v1"
# model_kwargs = {
#     "model": "claude-3-5-sonnet-20241022",
#     "temperature": 0.7,
#     "max_tokens": 500,
# }

# RUNPOD
# runpod_serverless_id = os.getenv("RUNPOD_SERVERLESS_ID")
# endpoint_url = f"https://api.runpod.ai/v2/{runpod_serverless_id}/openai/v1"
# model_kwargs = {
#     "model": "mistralai/Mistral-7B-Instruct-v0.3",
#     "temperature": 0.3,
#     "max_tokens": 500
# }

# Initialize clients
wcd_url = os.environ["WCD_URL"]
wcd_api_key = os.environ["WCD_API_KEY"]

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,
    auth_credentials=Auth.api_key(wcd_api_key),
    skip_init_checks=True
)

openai_client = wrap_openai(openai.AsyncClient(api_key=api_key, base_url=endpoint_url))

@cl.on_message
@traceable
async def on_message(message: cl.Message):
    # Maintain an array of messages in the user session
    message_history = cl.user_session.get("message_history", [])

    # Initialize message history with system prompt if empty
    if not message_history and ENABLE_SYSTEM_PROMPT:
        message_history.append({"role": "system", "content": SYSTEM_PROMPT})

    # Get the user's query
    query = message.content
    
    response_message = cl.Message(content="")
    await response_message.send()

    prompt = generate_prompt(query, message_history)
        
    # Update message history with the RAG prompt
    message_history.append({"role": "user", "content": prompt})

    # 5. Generate answer using GPT-4
    stream = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a Tesla Cybertruck assistant. You have access to the Tesla Cybertruck manual, if needed."},
            {"role": "user", "content": prompt}
        ],
        stream=True,
        **model_kwargs
    )
    
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()

    # Record the AI's response in the history
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)
    
async def generate_prompt(query, message_history):
    # quick router
    needs_context = await query_needs_context(query, message_history)
    
    if needs_context:
        prompt = await generate_rag_prompt(query, message_history)
    else:
        prompt = generate_quick_prompt(query, message_history)
    
    return prompt

"""
Determine if the user's query needs context from the embedded data to be answered accurately.

@param user_message: The user's query
@param chat_history: The chat history
@param client: The client to use for generating the response

"""
async def query_needs_context(user_message, chat_history):
    prompt_template = """
    Your job is to classify if the query can be answered with the information in the chat history. If the query can be answered with the information in the chat history, OR if the query is not a direct question about the Tesla Cybertruck, respond with NO. If the query could benefit with additional information from the Tesla Cybertruck manual, respond with YES.
    Only respond with YES or NO.
    
    Chat History:
    {chat_history}
    
    Current Query: {user_message}
    
    Does this query need additional context? (YES/NO):
    """
    router_prompt = prompt_template.format(chat_history=chat_history, user_message=user_message)
    
    response = await openai_client.chat.completions.create(messages=[{"role": "user", "content": router_prompt}], **smol_model_kwargs)
    
    print("Router Prompt: ", router_prompt)
    print("Response: ", response.choices[0].message.content)
    return response.choices[0].message.content.strip().upper() == "YES"


async def generate_rag_prompt(user_message, chat_history):
    search_query = await generate_search_query(user_message, chat_history)
    context = await retrieve_context(search_query)
    
    full_prompt = f"""
    Use the following context to answer the query. If the query is not related to the Tesla Cybertruck, respond with "I'm sorry, I don't know about that. If you use the context provided respond with 'I found the following information in the Tesla Cybertruck manual:'.
    
    Context: {context}
    
    User Query: {user_message}
    """

    print("RAG Prompt: ", full_prompt)
    return full_prompt

"""
Generate a quick response to the user's query.
@param user_message: The user's query
@param chat_history: The chat history
"""
def generate_quick_prompt(user_message, chat_history):
    quick_prompt = f"""
    You are a Cybertruck assistant. Answer the following query based on the information in the chat history.

    User Query: {user_message}
"""
    
    print("Quick Prompt: ", quick_prompt)
    return quick_prompt


"""
Generate a search query for the user's query.
@param user_message: The user's query
@param chat_history: The chat history
@param client: The client to use for generating the response
"""
async def generate_search_query(user_message, chat_history):
    prompt_template = """
    Based on the conversation history and current query, 
    generate a clear, standalone search query for searching the Cybertruck manual.
    
    Chat History:
    {chat_history}
    
    Current Query: {user_message}
    
    Generated Search Query:
    """
    
    context_prompt = prompt_template.format(chat_history=chat_history, user_message=user_message)
    print("Context Prompt: ", context_prompt)
    
    query = await openai_client.chat.completions.create(messages=[{"role": "user", "content": context_prompt}], **smol_model_kwargs)
    
    print("Query: ", query.choices[0].message.content)
    return query.choices[0].message.content.strip()


async def retrieve_context(search_query):
    print("Retrieving Context for: ", search_query)
    # 1. Generate embedding for the query
    embedding_response = await openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=search_query
    )
    query_embedding = embedding_response.data[0].embedding

    # 2. Retrieve relevant chunks from Weaviate
    collection = weaviate_client.collections.get("TeslaCybertruck")
    similar_texts = collection.query.near_vector(
        near_vector=query_embedding,
        limit=3,
        return_properties=["text"],
        return_metadata=MetadataQuery(distance=True)
    )
    # 3. Combine retrieved contexts
    context_str = "\n\n---\n\n".join(
        [doc.properties["text"] for doc in similar_texts.objects]
    )
    
    print("RetrievedContext: ", context_str)
    return context_str

@cl.on_shutdown
async def shutdown():
    weaviate_client.close()
    await openai_client.close()
