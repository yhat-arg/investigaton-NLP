"""
Run evaluation on Investigathon LLMTrack Evaluation dataset.

This script evaluates the model on the evaluation set which includes ground truth answers.
Uses the same RAG configuration as main.py.

Usage:
    python run_evaluation.py --dataset_type <oracle|full> --n_samples <number>
    
Example:
    python run_evaluation.py --dataset_type full --n_samples 250
"""

import argparse
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from src.models.LiteLLMModel import LiteLLMModel
from src.agents.judge.JudgeAgent import JudgeAgent
from src.agents.rag.RAGAgent import RAGAgent
from config.config import Config, MemoryModelConfig

load_dotenv()


class InvestigathonEvalDataset:
    """Dataset loader for Investigathon LLMTrack Evaluation data."""
    
    def __init__(self, dataset_type: str):
        paths = {
            "oracle": "data/investigathon/Investigathon_LLMTrack_Evaluation_oracle.json",  # Solo sesiones relevantes
            "full": "data/investigathon/Investigathon_LLMTrack_Evaluation_s_cleaned.json",  # Todas las sesiones (~115k tokens)
        }
        
        if dataset_type not in paths:
            raise ValueError(f"Invalid dataset type: {dataset_type}. Must be 'oracle' or 'full'")
        
        file_path = paths[dataset_type]
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}\n"
                f"Please run: python data/download_investigathon_data.py"
            )
        
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} instances from {file_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self.data[i] for i in range(*key.indices(len(self.data)))]
        else:
            return self.data[key]


def load_config_from_main() -> Config:
    """Load the same configuration used in main.py"""
    return Config(
        memory_model=MemoryModelConfig(
            model_type="litellm",
            model_name="ollama/gemma3:4b",
        ),
        memory_agent="RAG",
        judge_model_name="ollama/gemma3:4b",
        longmemeval_dataset_type="short",
        N=10,
    )


def format_instance_for_agent(instance: dict):
    """Format Investigathon instance to match LongMemEval format."""
    from src.datasets.LongMemEvalDataset import LongMemEvalInstance, Session
    
    # Convert sessions format
    sessions = []
    for session_id, date, messages in zip(
        instance.get("haystack_session_ids", []),
        instance.get("haystack_dates", []),
        instance.get("haystack_sessions", [])
    ):
        sessions.append(Session(session_id=session_id, date=date, messages=messages))
    
    return LongMemEvalInstance(
        question_id=instance["question_id"],
        question=instance["question"],
        sessions=sessions,
        t_question=instance["question_date"],
        answer=instance["answer"]
    )


def run_evaluation(
    dataset_type: str = "full",
    n_samples: int = 10,
):
    """Run evaluation on Investigathon dataset using the same config as main.py."""
    
    # Load config (same as main.py)
    config = load_config_from_main()
    
    # Load dataset
    dataset = InvestigathonEvalDataset(dataset_type)
    
    # Load models and agents
    print(f"\nInitializing models (using config from main.py)...")
    print(f"  Memory Model: {config.memory_model.model_name}")
    print(f"  Judge Model: {config.judge_model_name}")
    print(f"  Memory Agent: {config.memory_agent}")
    print(f"  Embedding Model: {config.embedding_model_name}")
    
    memory_model = LiteLLMModel(config.memory_model.model_name)
    judge_model = LiteLLMModel(config.judge_model_name)
    judge_agent = JudgeAgent(model=judge_model)
    memory_agent = RAGAgent(model=memory_model, embedding_model_name=config.embedding_model_name)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"data/results/investigathon_eval_{dataset_type}_{config.memory_agent}_{config.memory_model.model_name.replace('/', '_')}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nResults will be saved to: {results_dir}")
    print(f"Processing {n_samples} samples...")
    print("=" * 80)
    
    # Process samples
    correct_predictions = 0
    processed = 0
    
    for i, instance in enumerate(dataset[:n_samples]):
        question_id = instance["question_id"]
        result_file = f"{results_dir}/{question_id}.json"
        
        if os.path.exists(result_file):
            print(f"[{i+1}/{n_samples}] Skipping {question_id} (already exists)")
            continue
        
        try:
            # Format instance for agent
            formatted_instance = format_instance_for_agent(instance)
            
            # Get prediction
            predicted_answer = memory_agent.answer(formatted_instance)
            
            # Judge answer
            answer_is_correct = judge_agent.judge(formatted_instance, predicted_answer)
            
            if answer_is_correct:
                correct_predictions += 1
            
            # Save result
            result = {
                "question_id": question_id,
                "question": instance["question"],
                "predicted_answer": predicted_answer,
                "ground_truth": instance["answer"],
                "answer_is_correct": answer_is_correct,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            processed += 1
            
            # Print progress
            print(f"[{i+1}/{n_samples}] {question_id}")
            print(f"  Question: {instance['question'][:100]}...")
            print(f"  Predicted: {predicted_answer}")
            print(f"  Ground Truth: {instance['answer']}")
            print(f"  Correct: {answer_is_correct}")
            print(f"  Accuracy so far: {correct_predictions}/{processed} = {correct_predictions/processed*100:.1f}%")
            print("-" * 80)
            
        except Exception as e:
            print(f"[{i+1}/{n_samples}] Error processing {question_id}: {e}")
            print("-" * 80)
            continue
    
    # Final summary
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print(f"Correct predictions: {correct_predictions}/{processed}")
    if processed > 0:
        accuracy = correct_predictions / processed * 100
        print(f"Accuracy: {accuracy:.2f}%")
    
    # Save summary
    summary = {
        "dataset_type": dataset_type,
        "n_samples_requested": n_samples,
        "n_samples_processed": processed,
        "correct_predictions": correct_predictions,
        "accuracy": correct_predictions / processed if processed > 0 else 0,
        "memory_agent": config.memory_agent,
        "memory_model": config.memory_model.model_name,
        "judge_model": config.judge_model_name,
        "embedding_model": config.embedding_model_name,
        "timestamp": datetime.now().isoformat(),
        "results_dir": results_dir
    }
    
    with open(f"{results_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {results_dir}/summary.json")
    
    return accuracy if processed > 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on Investigathon LLMTrack dataset (uses config from main.py)"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="full",
        choices=["oracle", "full"],
        help="Type of evaluation dataset: 'oracle' (only relevant sessions) or 'full' (all ~53 sessions, ~115k tokens)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=250,
        help="Number of samples to process (default: 250)"
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        dataset_type=args.dataset_type,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()

