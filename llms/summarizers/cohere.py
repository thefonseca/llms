import os
from typing import List

import cohere

from .base import InstructTunedSummarizer
from ..utils.memoizer import memoize


def get_api_key():
    api_key = os.getenv("COHERE_API_KEY")
    if api_key is None:
        raise ValueError("Environment variable COHERE_API_KEY not set")
    return api_key


class CohereTokenizer:
    def __init__(self) -> None:
        # maximum number of characters supported by Cohere API
        self.max_characters = 65536

    @staticmethod
    @memoize()
    def encode_cached(text, max_characters):
        co = cohere.Client(get_api_key())
        text = text[:max_characters]
        return co.tokenize(text).tokens

    @staticmethod
    @memoize()
    def decode_cached(tokens: List[int]):
        co = cohere.Client(get_api_key())
        return co.detokenize(tokens).text

    def encode(self, text):
        return self.encode_cached(text, self.max_characters)

    def decode(self, tokens: List[int]):
        return self.decode_cached(tokens)


class CohereSummarizer(InstructTunedSummarizer):
    def __init__(self, model_name, request_interval=30, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        # wait an interval in seconds between requests
        self.request_interval = request_interval

    @staticmethod
    def load_tokenizer():
        return CohereTokenizer()

    def default_max_tokens(self):
        return 2048

    def default_user_prompt(self):
        return "{input}"

    def preprocess(self, text, truncation=True, **generation_kwargs):
        model_input, truncated_tokens, generation_kwargs = super().preprocess(
            text, truncation=truncation, **generation_kwargs
        )
        max_tokens = generation_kwargs.pop("max_length", 256)
        if max_tokens <= 50:
            generation_kwargs["length"] = "short"
        elif max_tokens <= 150:
            generation_kwargs["length"] = "medium"
        else:
            generation_kwargs["length"] = "long"

        format = generation_kwargs.pop("format", "paragraph")
        generation_kwargs["format"] = format
        extractiveness = generation_kwargs.pop("extractiveness", "high")
        generation_kwargs["extractiveness"] = extractiveness
        temperature = generation_kwargs.pop("temperature", 1.0)
        generation_kwargs["temperature"] = temperature
        model_input = "\n".join([m["content"] for m in model_input])
        return model_input, truncated_tokens, generation_kwargs

    @staticmethod
    @memoize()
    def generate_cached(
        model_name,
        model_input,
        memoizer_ignore_cache=False,
        **generation_kwargs,
    ):
        co = cohere.Client(get_api_key())
        response = co.summarize(model=model_name, text=model_input, **generation_kwargs)
        return response

    def postprocess(self, response):
        summary = response.summary
        summary = super().postprocess(summary)
        return summary
