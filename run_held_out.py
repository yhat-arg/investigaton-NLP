"""
Run predictions on Investigathon LLMTrack HeldOut dataset.

This script generates predictions for the held-out set which does NOT include ground truth answers.
Uses the same RAG configuration as main.py.
The output will be a JSON file with predictions that can be submitted for evaluation.

Usage:
    python run_held_out.py --n_samples <number>
    
Example:
    python run_held_out.py --n_samples 250
"""

import argparse
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from src.models.LiteLLMModel import LiteLLMModel
from src.agents.rag.RAGAgent import RAGAgent
from config.config import Config, MemoryModelConfig

load_dotenv()


class InvestigathonHeldOutDataset:
    """Dataset loader for Investigathon LLMTrack HeldOut data (without answers)."""
    
    def __init__(self):
        file_path = "data/investigathon/Investigathon_LLMTrack_HeldOut_s_cleaned.json"
        
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
    
    # Note: No answer field in held-out set
    return LongMemEvalInstance(
        question_id=instance["question_id"],
        question=instance["question"],
        sessions=sessions,
        t_question=instance["question_date"],
        answer=""  # Empty answer for held-out
    )


def run_held_out(
    n_samples: int = 250,
    output_file: str = None
):
    """Run predictions on HeldOut dataset using the same config as main.py and save results for submission."""
    
    # Load config (same as main.py)
    config = load_config_from_main()
    
    # Load dataset
    dataset = InvestigathonHeldOutDataset()
    
    # Load models and agents
    print(f"\nInitializing models (using config from main.py)...")
    print(f"  Memory Model: {config.memory_model.model_name}")
    print(f"  Memory Agent: {config.memory_agent}")
    print(f"  Embedding Model: {config.embedding_model_name}")
    
    memory_model = LiteLLMModel(config.memory_model.model_name)
    memory_agent = RAGAgent(model=memory_model, embedding_model_name=config.embedding_model_name)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file is None:
        output_file = f"data/results/investigathon_heldout_{config.memory_agent}_{config.memory_model.model_name.replace('/', '_')}_{timestamp}.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\nResults will be saved to: {output_file}")
    print(f"Processing {n_samples} samples...")
    print("=" * 80)
    
    # Process samples and collect predictions
    predictions = []
    processed = 0
    
    for i, instance in enumerate(dataset[:n_samples]):
        question_id = instance["question_id"]
        
        try:
            # Format instance for agent
            formatted_instance = format_instance_for_agent(instance)
            
            # Get prediction
            print(f"[{i+1}/{n_samples}] Processing {question_id}...", flush=True)
            predicted_answer = memory_agent.answer(formatted_instance)
            
            # Store prediction
            predictions.append({
                "question_id": question_id,
                "predicted_answer": predicted_answer
            })
            
            processed += 1
            
            # Print progress
            print(f"  Question: {instance['question'][:100]}...")
            print(f"  Predicted: {predicted_answer}")
            print("-" * 80)
            
        except Exception as e:
            print(f"[{i+1}/{n_samples}] Error processing {question_id}: {e}")
            print("-" * 80)
            # Add empty prediction for failed cases
            predictions.append({
                "question_id": question_id,
                "predicted_answer": "ERROR: Unable to generate prediction",
                "error": str(e)
            })
            continue
    
    # Save predictions
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    # Final summary
    print("=" * 80)
    print("HELD-OUT PREDICTIONS COMPLETE")
    print(f"Processed: {processed}/{n_samples}")
    print(f"Predictions saved to: {output_file}")
    print("\nThis file is ready for submission!")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for Investigathon LLMTrack HeldOut dataset (uses config from main.py)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=250,
        help="Number of samples to process (default: 250 - all held-out instances)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path for predictions (optional)"
    )
    
    args = parser.parse_args()
    
    run_held_out(
        n_samples=args.n_samples,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()

