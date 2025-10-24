import re
import json
import torch
from typing import Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.models.TransformersModel import TransformersModel


class QwenModel(TransformersModel):
    def parse_tool_calls(self, content: str) -> list:
        tool_calls = []
        pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        matches = re.findall(pattern, content, re.DOTALL)

        for i, match in enumerate(matches):
            call_data = json.loads(match)
            tool_calls.append(
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": call_data["name"],
                        "arguments": json.dumps(call_data["arguments"]),
                    },
                }
            )

        return tool_calls

    def parse_thinking(self, content: str) -> tuple[Optional[str], str]:
        thinking_blocks = [
            block.strip()
            for block in re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
            if block.strip()
        ]

        cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        reasoning_content = "\n\n".join(thinking_blocks) if thinking_blocks else None

        return reasoning_content, cleaned_content
