from typing import Literal
from pydantic import BaseModel, Field


class MemoryModelConfig(BaseModel):
    """Configuration for memory models."""

    model_type: Literal["transformers", "litellm"] = Field(
        ..., description="Type of memory model to use"
    )
    model_name: str = Field(..., description="Name of the model")
    quantized: bool = Field(
        default=False, description="Whether to use quantization for transformers models"
    )

class Config(BaseModel):
    """Configuration class for LongMemEval experiments."""

    memory_model: MemoryModelConfig = Field(..., description="Memory model configuration")
    embedding_model_name: str = Field(
        default="ollama/nomic-embed-text", description="Name of the embedding model"
    )
    judge_model_name: str = Field(..., description="Judge model name")

    memory_agent: Literal["RAG", "FullContext"] = Field(..., description="Memory agent to use")

    longmemeval_dataset_type: Literal["oracle", "short", "long"] = Field(
        ..., description="Type of LongMemEval dataset to use"
    )

    N: int = Field(default=10, description="Number of samples to process")

    class Config:
        """Pydantic configuration."""

        extra = "forbid"  # Prevent extra fields
        validate_assignment = True  # Validate on assignment
