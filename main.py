from dotenv import load_dotenv
from src.experiments.pass_whole_context.longmemeval_experiment import run_experiment
from src.core.Config import Config, MemoryModelConfig


load_dotenv()

if __name__ == "__main__":
    config = Config(
        memory_model=MemoryModelConfig(
            model_type="transformers",
            model_name="Qwen/Qwen3-4B",
        ),
        memory_agent="RAG",
        judge_model_name="azure/gpt-4.1",
        longmemeval_dataset_type="short",
        N=-1,
    )
    run_experiment(config)
