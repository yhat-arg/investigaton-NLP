import json
import os
from dotenv import load_dotenv
from src.models.LiteLLMModel import LiteLLMModel
from src.agents.judge.JudgeAgent import JudgeAgent
from src.agents.rag.RAGAgent import RAGAgent
from src.datasets.LongMemEvalDataset import LongMemEvalDataset
from config.config import Config, MemoryModelConfig


load_dotenv()

if __name__ == "__main__":
    config = Config(
        memory_model=MemoryModelConfig(
            model_type="litellm",
            model_name="ollama/gemma3:4b",
        ),
        memory_agent="RAG",
        judge_model_name="ollama/gemma3:4b",
        longmemeval_dataset_type="short",
        N=-1,
    )
    
    # Initialize models and agents
    print(f"\nInitializing models...")
    print(f"  Memory Model: {config.memory_model.model_name}")
    print(f"  Judge Model: {config.judge_model_name}")
    print(f"  Memory Agent: {config.memory_agent}")
    print(f"  Embedding Model: {config.embedding_model_name}")
    
    memory_model = LiteLLMModel(config.memory_model.model_name)
    judge_model = LiteLLMModel(config.judge_model_name)
    judge_agent = JudgeAgent(model=judge_model)
    memory_agent = RAGAgent(model=memory_model, embedding_model_name=config.embedding_model_name)
    
    # Load dataset
    longmemeval_dataset = LongMemEvalDataset(config.longmemeval_dataset_type)
    
    # Create results directory
    results_dir = f"data/results/longmemeval_{config.longmemeval_dataset_type}_{config.memory_agent}_{config.memory_model.model_name.replace('/', '_')}_{config.judge_model_name.replace('/', '_')}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nResults will be saved to: {results_dir}")
    print(f"Processing samples...")
    print("=" * 100)
    
    # Process samples
    correct_predictions = 0
    processed = 0
    
    for instance in longmemeval_dataset[:config.N]:
        if os.path.exists(f"{results_dir}/{instance.question_id}.json"):
            print(f"Skipping {instance.question_id} because it already exists", flush=True)
            continue
        
        try:
            predicted_answer = memory_agent.answer(instance)
            answer_is_correct = judge_agent.judge(instance, predicted_answer)
            
            if answer_is_correct:
                correct_predictions += 1
            
            processed += 1
            
            # Save result
            with open(f"{results_dir}/{instance.question_id}.json", "w") as f:
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
    
    # Final summary
    print("=" * 100)
    if processed > 0:
        accuracy = correct_predictions / processed
        print(f"Correct predictions: {correct_predictions}/{processed}")
        print(f"Accuracy: {accuracy:.2%}")
    else:
        print("No samples were processed.")


# Otros ejemplos de configuracion
# Si pasamos esta MemoryModelConfig, vamos a usar un modelo de transformers de huggingface.
# MemoryModelConfig(
#     model_type="transformers",
#     model_name="Gemma/gemma3:4b",
# ),
# Para que esto funcione, necesitan correrlo en una maquin acorde.

# Tambien podemos usar un modelo con mucho contexto para probar.
# Otro detalle, para los modelos de openai, se puede especificar el proveedor.
# config = Config(
#     memory_model=MemoryModelConfig(
#         model_type="litellm",
#         model_name="openai/gpt-5",
#     ),
#     memory_agent="RAG",
#     judge_model_name="openai/gpt-5",
#     longmemeval_dataset_type="short",
#     N=-1,
# )
