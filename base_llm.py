from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, base_ckpt="HuggingFaceTB/SmolLM2-360M-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
        self.model = AutoModelForCausalLM.from_pretrained(base_ckpt).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        return question

    def parse_answer(self, answer: str) -> str:
        """
        # Change: Previously returned a float, but now returns a string (str).
        # Chatbot responses are in text form, not numbers, so the parsing method is changed.
        # Previous code tried to extract numbers, but now we return the full text response.
        """
        import re
        
        # Check if the answer contains <answer> tags
        if "<answer>" in answer:
            # Find the last occurrence of <answer> tag
            last_answer_pos = answer.rfind("<answer>")
            
            # Extract content after the last <answer> tag
            content_after_tag = answer[last_answer_pos + len("<answer>"):]
            
            # Check if there's a closing </answer> tag after the last opening tag
            if "</answer>" in content_after_tag:
                end_pos = content_after_tag.find("</answer>")
                return content_after_tag[:end_pos].strip()
            else:
                # If no closing tag, return everything after the last <answer>
                return content_after_tag.strip()
        else:
            return answer.strip() if answer else ""

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate` method.

        You will likely get an up to 10x speedup using batched decoding.

        To implement batch decoding you will need to:
        - tokenize the prompts self.tokenizer with padding=True and return_tensors="pt"
        - call self.model.generate
        - decode the outputs with self.tokenizer.batch_decode

        """
        from tqdm import tqdm  # Importing tqdm for progress bar

        # Preventing OOM
        # Depending on your GPU batched generation will use a lot of memory.
        # If you run out of memory, try to reduce the micro_batch_size.
        micro_batch_size = 16
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
                )
                for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
            ]

        # Set padding side to left for proper alignment during generation
        self.tokenizer.padding_side = "left"
        
        # Tokenize the prompts with padding
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Set up generation parameters
        generation_kwargs = {
            "max_new_tokens": 256,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Add temperature and sampling parameters if temperature > 0
        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        
        # Add num_return_sequences if specified
        if num_return_sequences is not None:
            generation_kwargs["num_return_sequences"] = num_return_sequences
        
        # Generate outputs
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generation_kwargs
        )
        
        # Decode the outputs
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # If num_return_sequences is specified, reshape the output
        if num_return_sequences is not None and num_return_sequences > 1:
            # Reshape to list of lists, where each inner list contains generations for one prompt
            result = []
            for i in range(0, len(decoded_outputs), num_return_sequences):
                result.append(decoded_outputs[i:i + num_return_sequences])
            return result
        else:
            return decoded_outputs

    def answer(self, *questions) -> list[str]:
        """
        Answer questions given as individual string arguments.
        # Change: Return type changed from list[float] to list[str].
        # Chatbot responses are text, not numbers, so the return type is changed to a list of strings.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        print("generations:", generations)
        return [self.parse_answer(g) for g in generations]



if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
