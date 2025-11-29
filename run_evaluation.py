import json
import os
from dotenv import load_dotenv
from src.models.LiteLLMModel import LiteLLMModel
from src.agents.judge.JudgeAgent import JudgeAgent
from src.agents.rag.RAGAgent import RAGAgent
from src.datasets.LongMemEvalDataset import LongMemEvalDataset
from config.config import Config

load_dotenv()

config = Config(
    memory_model_name="ollama/gemma3:4b",
    judge_model_name="ollama/gemma3:4b",
    longmemeval_dataset_type="oracle",
    longmemeval_dataset_set="investigathon_evaluation",
    N=10,
)

print(f"\nInitializing models (using config from main.py)...")
print(f"  Memory Model: {config.memory_model_name}")
print(f"  Judge Model: {config.judge_model_name}")
print(f"  Embedding Model: {config.embedding_model_name}")

memory_model = LiteLLMModel(config.memory_model_name)
judge_model = LiteLLMModel(config.judge_model_name)
judge_agent = JudgeAgent(model=judge_model)
memory_agent = RAGAgent(model=memory_model, embedding_model_name=config.embedding_model_name)

# Load dataset
dataset = LongMemEvalDataset(config.longmemeval_dataset_type, config.longmemeval_dataset_set)

# Create results directory
results_dir = f"data/results/investigathon_eval_{config.longmemeval_dataset_set}/{config.longmemeval_dataset_type}_embeddings_{config.embedding_model_name.replace('/', '_')}_memory_{config.memory_model_name.replace('/', '_')}_judge_{config.judge_model_name.replace('/', '_')}"
os.makedirs(results_dir, exist_ok=True)

print(f"\nResults will be saved to: {results_dir}")
print(f"Processing {config.N} samples...")
print("=" * 80)

# Process samples
for i, instance in enumerate(dataset[: config.N]):
    result_file = f"{results_dir}/{instance.question_id}.json"

    if os.path.exists(result_file):
        print(f"[{i+1}/{config.N}] Skipping {instance.question_id} (already exists)")
        continue

    predicted_answer = memory_agent.answer(instance)
    answer_is_correct = judge_agent.judge(instance, predicted_answer)

    # Save result
    with open(result_file, "w", encoding="utf-8") as f:
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

    print(f"  Question: {instance.question}...")
    print(f"  Predicted: {predicted_answer}")
    print(f"  Ground Truth: {instance.answer}")
    print(f"  Correct: {answer_is_correct}")
    print("-" * 100)

print("=" * 80)
print("EVALUATION COMPLETE")
