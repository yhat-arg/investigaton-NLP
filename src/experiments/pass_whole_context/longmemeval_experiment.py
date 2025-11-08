import json
import os
import pandas as pd
from src.models.LiteLLMModel import LiteLLMModel
from src.agents.judge.JudgeAgent import JudgeAgent
from src.datasets.LongMemEvalDataset import LongMemEvalDataset
from config.config import Config
from src.models.Model import Model
from src.models.QwenModel import QwenModel
from src.agents.rag.RAGAgent import RAGAgent
from src.agents.full_context.FullContextAgent import FullContextAgent


def load_memory_model(config: Config):
    if config.memory_model.model_type == "transformers":
        return QwenModel(config.memory_model.model_name, quantized=config.memory_model.quantized)
    elif config.memory_model.model_type == "litellm":
        return LiteLLMModel(config.memory_model.model_name)
    else:
        raise ValueError(f"Invalid model type: {config.memory_model.model_type}")


def load_memory_agent(memory_model: Model, config: Config):
    if config.memory_agent == "RAG":
        return RAGAgent(model=memory_model)
    elif config.memory_agent == "FullContext":
        return FullContextAgent(model=memory_model)
    else:
        raise ValueError(f"Invalid memory agent: {config.memory_agent}")


def run_experiment(config: Config):
    memory_model = load_memory_model(config)

    judge_model = LiteLLMModel(config.judge_model_name)
    judge_agent = JudgeAgent(model=judge_model)

    memory_agent = load_memory_agent(memory_model, config)

    correct_predictions = 0
    longmemeval_dataset = LongMemEvalDataset(config.longmemeval_dataset_type)

    results_dir = f"data/results/longmemeval_{config.longmemeval_dataset_type}_{config.memory_agent}_{config.memory_model.model_name.replace("/", "_")}_{config.judge_model_name.replace("/", "_")}"
    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    for instance in longmemeval_dataset[: config.N]:
        if os.path.exists(f"{results_dir}/{instance.question_id}.json"):
            print(f"Skipping {instance.question_id} because it already exists", flush=True)
            continue
        try:
            predicted_answer = memory_agent.answer(instance)
            answer_is_correct = judge_agent.judge(instance, predicted_answer)
            if answer_is_correct:
                correct_predictions += 1

            with open(
                f"{results_dir}/{instance.question_id}.json",
                "w",
            ) as f:
                json.dump(
                    {
                        "question_id": instance.question_id,
                        "question": instance.question,
                        "predicted_answer": predicted_answer,
                        "answer": instance.answer,
                        "answer_is_correct": answer_is_correct,
                    },
                    f,
                    indent=2,
                )

            print(f"Question: {instance.question}", flush=True)
            print(f"Predicted Answer: {predicted_answer}", flush=True)
            print(f"Answer: {instance.answer}", flush=True)
            print(f"Correct: {answer_is_correct}", flush=True)
            print("-" * 100, flush=True)
        except Exception as e:
            print(f"Error: {e}", flush=True)
            print("-" * 100, flush=True)
            continue

    print(f"Correct predictions: {correct_predictions}/{config.N}")
    print(f"Accuracy: {correct_predictions/config.N}")

    return correct_predictions / config.N
