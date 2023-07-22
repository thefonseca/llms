import os

import openai
import tiktoken

from .base import PromptBasedLM
from ..utils.memoizer import memoize

openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIChat(PromptBasedLM):
    def __init__(self, model_name, request_interval=30, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        self.tokenizer = self.load_tokenizer()
        # wait an interval in seconds between requests
        self.request_interval = request_interval

    def load_tokenizer(self):
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return encoding

    def num_tokens_for_prompt(self, messages):
        """Returns the number of tokens used by a list of messages."""

        # note: future models may deviate from this
        if "gpt-" in self.model_name:
            num_tokens = 0
            for message in messages:
                # every message follows <im_start>{role/name}\n{content}<im_end>\n
                num_tokens += 4
                for key, value in message.items():
                    num_tokens += len(self.tokenizer.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens
        else:
            raise NotImplementedError(
                f"""num_tokens_for_prompt() is not presently implemented for model {self.model_name}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )

    def default_max_tokens(self):
        return 4096

    @staticmethod
    @memoize()
    def generate_cached(
        model_name,
        model_input,
        memoizer_ignore_cache=False,
        **generation_kwargs,
    ):
        response = openai.ChatCompletion.create(
            model=model_name, messages=model_input, **generation_kwargs
        )
        return response

    def preprocess(self, text, truncation=True, max_length=None, **generation_kwargs):
        prompt, truncated_tokens, generation_kwargs = super().preprocess(
            text, truncation=truncation, max_length=max_length, **generation_kwargs
        )
        generation_kwargs["max_tokens"] = max_length
        generation_kwargs.pop("seed", None)
        return prompt, truncated_tokens, generation_kwargs

    def postprocess(self, output):
        output = output["choices"][0]["message"]["content"]
        output = super().postprocess(output)
        return output
