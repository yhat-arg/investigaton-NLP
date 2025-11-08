import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.agents.MemoryAgent import MemoryAgent
from src.datasets.LongMemEvalDataset import LongMemEvalInstance
from litellm import embedding


def embed_text(message):
    # NOTE: This is using azure embeddings. If you want, you can change it to openai
    response = embedding(model="azure/text-embedding-3-small", input=message)
    return response.data[0]["embedding"]


def get_messages_and_embeddings(instance: LongMemEvalInstance):

    cache_path = f"data/rag/embeddings/{instance.question_id}.parquet"
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        return df["messages"].tolist(), df["embeddings"].tolist()

    messages = []
    embeddings = []
    for session in tqdm(instance.sessions, desc="Embedding sessions"):
        for message in session.messages:
            messages.append(message)
            embeddings.append(embed_text(f"{message['role']}: {message['content']}"))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    pd.DataFrame({"messages": messages, "embeddings": embeddings}).to_parquet(cache_path)
    return messages, embeddings


def retrieve_most_relevant_messages(instance: LongMemEvalInstance, k: int):

    question_embedding = embed_text(instance.question)
    messages, embeddings = get_messages_and_embeddings(instance)

    similarity_scores = np.dot(embeddings, question_embedding)
    most_relevant_messages_indices = np.argsort(similarity_scores)[::-1][:k]
    most_relevant_messages = [messages[i] for i in most_relevant_messages_indices]

    return most_relevant_messages


class RAGAgent(MemoryAgent):
    def __init__(self, model="azure/gpt-4.1"):
        self.model = model

    def answer(self, instance: LongMemEvalInstance):
        most_relevant_messages = retrieve_most_relevant_messages(instance, 10)

        prompt = f"""
        You are a helpful assistant that answers a question based on the evidence.
        The evidence is: {most_relevant_messages}
        The question is: {instance.question}
        Return the answer to the question.
        """
        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages)
        return answer
