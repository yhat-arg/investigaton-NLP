import torch
from typing import Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class TransformersModel:
    def __init__(
        self,
        name,
        quantized: bool = False,
    ) -> None:
        self.name = name
        self.quantized = quantized
        self.tokenizer, self.local_model = self.load_base_model(quantized=quantized)

    def load_base_model(self, quantized: bool = False):
        if quantized:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.name, device_map="auto", quantization_config=bnb_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.name, device_map="auto", dtype=torch.float16
            )

        tokenizer = AutoTokenizer.from_pretrained(self.name)
        return tokenizer, model

    def parse_tool_calls(self, content: str) -> list:
        raise NotImplementedError("Not implemented")

    def parse_thinking(self, content: str) -> tuple[Optional[str], str]:
        raise NotImplementedError("Not implemented")

    def reply(self, messages, tools):
        formatted_input = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(formatted_input, return_tensors="pt").to("cuda")
        outputs = self.local_model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        reasoning_content, cleaned_response_text = self.parse_thinking(response_text)

        tool_calls = self.parse_tool_calls(cleaned_response_text)

        return reasoning_content, tool_calls