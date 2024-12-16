import chainlit as cl
from langsmith import traceable
from prompts import SYSTEM_PROMPT

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

from clients import weaviate_client, openai_client, model_kwargs
from request import generate_prompt

from dotenv import load_dotenv

load_dotenv()

ENABLE_SYSTEM_PROMPT=True

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

    prompt = await generate_prompt(query, message_history)
        
    # Update message history with the RAG prompt
    message_history.append({"role": "user", "content": prompt})

    # 5. Generate answer using GPT-4
    stream = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
    

@cl.on_shutdown
async def shutdown():
    weaviate_client.close()
    await openai_client.close()
