import logging
from pprint import pformat

import numpy as np

from ..utils.utils import sent_tokenize, get_progress_bar, add_progress_task, log

logger = logging.getLogger(__name__)


class BaseLM:
    def __init__(self, model_name) -> None:
        self.model_name = model_name

    def __repr__(self):
        attr_dict = self.__dict__.copy()
        attr_dict["default_max_tokens"] = self.default_max_tokens()
        return f"{self.__class__.__name__}:\n{pformat(attr_dict)}"

    def load_tokenizer(**model_kwargs):
        raise NotImplementedError("Method load_tokenizer not implemented.")

    def default_max_tokens(self):
        """Return the maximum supported sequence length in tokens."""
        raise NotImplementedError("Function default_max_tokens not implemented.")

    def generate_cached(
        self,
        model_name,
        model_input,
        memoizer_ignore_cache=False,
        **generation_kwargs,
    ):
        raise NotImplementedError("Method generate_cached not implemented.")

    def get_model_kwargs(self):
        return {}

    def get_tokenizer_kwargs(self):
        return {}

    def truncate_input(self, input, max_tokens):
        tokenizer = self.load_tokenizer()
        input = tokenizer.encode(
            input, max_length=max_tokens, truncation=True, padding="max_length"
        )
        input = tokenizer.decode(
            input, skip_special_tokens=True, clean_up_tokenization_spaces=None
        )
        return input

    def cached_generation_fn(self):
        return self.generate_cached

    def build_input(self, input_data, verbose=False, **kwargs):
        return input_data, kwargs

    def preprocess(
        self, input_data, truncation=True, verbose=False, **generation_kwargs
    ):
        if isinstance(input_data, list):
            input_data = "\n".join(input_data)
        model_input, generation_kwargs = self.build_input(
            input_data, verbose=verbose, **generation_kwargs
        )
        truncated_tokens = 0
        if truncation:
            max_tokens = generation_kwargs.get("max_tokens", self.default_max_tokens())
            log(logger, f"Truncating input to {max_tokens} max tokens", verbose=verbose)
            model_input, truncated_tokens = self.truncate_input(model_input, max_tokens)

        return model_input, truncated_tokens, generation_kwargs

    def postprocess(self, output):
        output = "\n".join([s.strip() for s in sent_tokenize(output)])
        return output

    def generate(
        self,
        input_data,
        truncation=True,
        verbose=False,
        **generation_kwargs,
    ):
        model_input, _, generation_kwargs = self.preprocess(
            input_data, truncation=truncation, verbose=verbose, **generation_kwargs
        )
        model_kwargs = self.get_model_kwargs()
        kwargs = model_kwargs.copy()
        log(logger, f"Model kwargs:\n{pformat(kwargs)}", verbose=verbose)
        gen_kwargs_str = f"Generation kwargs:\n{pformat(generation_kwargs)}"
        log(logger, gen_kwargs_str, verbose=verbose)
        kwargs.update(generation_kwargs)
        log(logger, f"Model input:\n{pformat(model_input)}", verbose=verbose)
        output = self.generate_cached(self.model_name, model_input, **kwargs)
        output = self.postprocess(output)
        log(logger, f"Output:\n{pformat(output)}", verbose=verbose, max_length=None)
        return output

    def is_last_result_from_cache(self):
        is_cache_hit = hasattr(
            self.cached_generation_fn(), "memoize_info"
        ) and self.cached_generation_fn().memoize_info.get("cache_hit")
        return is_cache_hit


class PromptBasedLM(BaseLM):
    def __init__(
        self,
        model_name,
        system_prompt=None,
        user_prompt=None,
        **kwargs,
    ) -> None:
        super().__init__(model_name, **kwargs)
        if user_prompt is None:
            user_prompt = self.default_user_prompt()
        if system_prompt is None:
            system_prompt = self.default_system_prompt()
        self.user_prompt = user_prompt
        self.system_prompt = system_prompt

    def default_system_prompt(self):
        return None

    def default_user_prompt(self):
        return "{input}"

    def get_prompt_args(self):
        return {}

    def preprocess(
        self, input_data, truncation=True, max_length=None, **generation_kwargs
    ):
        max_tokens = generation_kwargs.pop("max_tokens", self.default_max_tokens())
        if max_length:
            max_tokens = max_tokens - max_length - 1
        prompt, truncated_tokens, generation_kwargs = super().preprocess(
            input_data,
            truncation=truncation,
            max_tokens=max_tokens,
            **generation_kwargs,
        )
        return prompt, truncated_tokens, generation_kwargs

    def truncate_input(self, prompt, max_tokens):
        # discount maximum output length from max_tokens
        num_tokens = self.num_tokens_for_prompt(prompt)
        excess_tokens = num_tokens - max_tokens
        # choose the longest message for truncation
        message_lengths = [len(m["content"]) for m in prompt]
        longest_idx = np.argmax(message_lengths)

        if excess_tokens > 0:
            tokenizer = self.load_tokenizer()
            truncated_prompt = [dict(item) for item in prompt]
            text = prompt[longest_idx]["content"]
            tokens = tokenizer.encode(text)
            tokens = tokens[:-excess_tokens]
            try:
                truncated_prompt[longest_idx]["content"] = tokenizer.decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=None
                )
            except TypeError:
                truncated_prompt[longest_idx]["content"] = tokenizer.decode(tokens)

            prompt = truncated_prompt
        return prompt, excess_tokens

    def build_input(
        self,
        input_data,
        system_prompt=None,
        user_prompt=None,
        budget=6,
        budget_unit="sentences",
        verbose=False,
        **generation_kwargs,
    ):
        if user_prompt is None:
            user_prompt = self.user_prompt
        if system_prompt is None:
            system_prompt = self.system_prompt

        # a naive way of converting unit to singular...
        if budget == 1 and budget_unit[-1].lower() == "s":
            budget_unit = budget_unit[:-1]

        prompt_args = dict(
            budget=budget,
            budget_unit=budget_unit,
            **generation_kwargs,
        )
        prompt_args = dict(prompt_args, **self.get_prompt_args())

        if isinstance(input_data, dict):
            prompt_args = dict(prompt_args, **input_data)
        else:
            prompt_args["input"] = input_data

        prompt = []

        if system_prompt:
            system_prompt = system_prompt.format(**prompt_args)
            prompt.append({"role": "system", "content": system_prompt})

        if user_prompt:
            user_prompt = user_prompt.format(**prompt_args)
            prompt.append({"role": "user", "content": user_prompt})

        log(logger, f"System prompt: {pformat(system_prompt)}", verbose=verbose)
        log(logger, f"User prompt: {pformat(user_prompt)}", verbose=verbose)
        return prompt, generation_kwargs

    def num_tokens_for_prompt(self, messages):
        tokenizer = self.load_tokenizer()
        # account for one newline after each message
        newline_tokens = tokenizer.encode("\n")
        num_tokens = len(newline_tokens) * (len(messages) - 1)
        for message in messages:
            num_tokens += len(tokenizer.encode(message["content"]))
        return num_tokens

    def token_statistics_for_input(self, input_data, truncation, **generation_kwargs):
        prompt, truncated_tokens, _ = self.preprocess(
            input_data, truncation=truncation, **generation_kwargs
        )
        tokenizer = self.load_tokenizer()
        if not isinstance(prompt, str):
            prompt = self.prompt_to_text(prompt)
        num_tokens = len(tokenizer.encode(prompt))
        return num_tokens, truncated_tokens

    def token_statistics(
        self,
        inputs,
        truncation=True,
        **generation_kwargs,
    ):
        progress = get_progress_bar()
        task = add_progress_task(
            progress,
            f"Calculating token statistics for {self.model_name}...",
            total=len(inputs),
            existing_ok=False,
        )
        truncated_tokens = []
        num_tokens = []

        with progress:
            for input_data in inputs:
                result = self.token_statistics_for_input(
                    input_data, truncation, **generation_kwargs
                )
                num_tokens.append(result[0])
                truncated_tokens.append(result[1])
                progress.update(task, advance=1)

        stats = dict(
            total_tokens=sum(num_tokens),
            mean_tokens=np.mean(num_tokens),
            total_truncation=sum(truncated_tokens),
            mean_truncation=np.mean(truncated_tokens),
        )
        return stats

    def prompt_to_text(self, prompt):
        return "\n".join([m["content"] for m in prompt])
