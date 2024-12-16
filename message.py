def get_answer(query: str, 
               weaviate_client, 
               openai_client, 
               collection_name: str = "TeslaCybertruck",
               num_chunks: int = 3) -> str:
    """
    Get an answer to a question using RAG (Retrieval Augmented Generation).
    
    Args:
        query (str): The user's question
        weaviate_client: Initialized Weaviate client
        openai_client: Initialized OpenAI client
        collection_name (str): Name of the Weaviate collection to query
        num_chunks (int): Number of relevant chunks to retrieve
        
    Returns:
        str: The generated answer
    """
    # 1. Generate embedding for the query
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    query_embedding = response.data[0].embedding

    # 2. Retrieve relevant chunks from Weaviate
    collection = weaviate_client.collections.get(collection_name)
    similar_texts = collection.query.near_vector(
        near_vector=query_embedding,
        limit=num_chunks,
        return_properties=["text"],
        return_metadata=MetadataQuery(distance=True)
    )

    # 3. Combine retrieved contexts
    context_str = "\n\n---\n\n".join(
        [doc.properties["text"] for doc in similar_texts.objects]
    )
    
    # 4. Create prompt for GPT
    prompt = f"""Answer the question using ONLY the information provided in the context below. 
    Do not add any general knowledge or information not contained in the context.

    Context:
    {context_str}

    Question: {query}

    Answer:"""

    # 5. Generate answer using GPT-4
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content
