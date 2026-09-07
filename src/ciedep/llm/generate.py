"""Hugging Face causal LM 실행기.

Table II 에서 비교한 모델군(Gemma, LLaMA, Qwen)을 chat template 로 동일하게 다룬다.
최종 모델은 Qwen2.5-7B-Instruct + CoT + self-consistency 를 사용한다.

인증이 필요한 모델은 환경변수 HF_TOKEN 을 읽는다. 토큰을 코드에 적지 않는다.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import torch

Message = Dict[str, str]


class LLMRunner:
    """chat template 기반 텍스트 생성기."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        token: Optional[str] = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        token = token or os.environ.get("HF_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            token=token,
        )
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        messages: Sequence[Message],
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """chat message 리스트를 받아 생성 결과 문자열 목록을 돌려준다."""
        inputs = self.tokenizer.apply_chat_template(
            list(messages),
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        input_length = inputs.shape[1]

        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if do_sample:
            # 다양한 추론 경로를 유도하기 위한 설정 (논문 III-B-2)
            generate_kwargs.update(temperature=temperature, top_p=top_p)

        outputs = self.model.generate(inputs, **generate_kwargs)
        new_tokens = outputs[:, input_length:]
        return [t.strip() for t in self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]

    def generate_one(self, messages: Sequence[Message], **kwargs) -> str:
        return self.generate(messages, **kwargs)[0]
