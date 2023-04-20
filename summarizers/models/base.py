import logging
from pprint import pformat

import numpy as np

from ..utils import sent_tokenize, get_progress_bar, add_progress_task, log

logger = logging.getLogger(__name__)


class Summarizer:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        
    def __repr__(self):
        attr_dict = self.__dict__.copy()
        attr_dict["default_max_tokens"] = self.default_max_tokens()
        return f"{self.__class__.__name__}:\n{pformat(attr_dict)}"

    def load_tokenizer(model_name_or_path, **model_kwargs):
        raise NotImplementedError("Method load_tokenizer not implemented.")

    def default_max_tokens(self):
        """Return the maximum supported sequence length in tokens."""
        raise NotImplementedError("Function default_max_tokens not implemented.")

    def generate_cached(
        self,
        model_name_or_path,
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
        tokenizer_kwargs = self.get_tokenizer_kwargs()
        tokenizer = self.load_tokenizer(self.model_name, **tokenizer_kwargs)
        input = tokenizer.encode(
            input, max_length=max_tokens, truncation=True, padding="max_length"
        )
        input = tokenizer.decode(
            input, skip_special_tokens=True, clean_up_tokenization_spaces=None
        )
        return input

    def cached_generation_fn(self):
        return self.generate_cached

    def build_input(self, text, **kwargs):
        return text

    def preprocess(self, text, truncation=True, verbose=False, **generation_kwargs):
        if isinstance(text, list):
            text = "\n".join(text)
        model_input, generation_kwargs = self.build_input(text, 
                                                          verbose=verbose, 
                                                          **generation_kwargs)

        if truncation:
            max_tokens = generation_kwargs.get(
                "max_tokens", self.default_max_tokens()
            )
            log(logger, f"Truncating input to {max_tokens} max tokens", verbose=verbose)
            model_input, truncated_tokens = self.truncate_input(model_input, max_tokens)

        return model_input, truncated_tokens, generation_kwargs
        
    def postprocess(self, summary):
        # rougeLSum expects newline after each sentence
        summary = "\n".join([s.strip() for s in sent_tokenize(summary)])
        return summary

    def generate(
        self,
        text,
        truncation=True,
        verbose=False,
        **generation_kwargs,
    ):
        model_input, _, generation_kwargs = self.preprocess(
            text, truncation=truncation, verbose=verbose, **generation_kwargs
        )
        model_kwargs = self.get_model_kwargs()
        kwargs = model_kwargs.copy()
        log(logger, f"Model kwargs:\n{pformat(kwargs)}", verbose=verbose)
        log(logger, f"Generation kwargs:\n{pformat(generation_kwargs)}", verbose=verbose)
        kwargs.update(generation_kwargs)
        log(logger, f"Model input:\n{pformat(model_input)}", verbose=verbose)
        summary = self.generate_cached(self.model_name, model_input, **kwargs)
        summary = self.postprocess(summary)
        log(logger, f"Summary:\n{pformat(summary)}", verbose=verbose, max_length=None)
        return summary

    def is_last_result_from_cache(self):
        is_cache_hit = hasattr(
            self.cached_generation_fn(), "memoize_info"
        ) and self.cached_generation_fn().memoize_info.get("cache_hit")
        return is_cache_hit


class PromptBasedSummarizer(Summarizer):
    def __init__(
        self,
        model_name,
        system_prompt=None,
        article_prompt=None,
        task_prompt=None,
        **kwargs,
    ) -> None:
        super().__init__(model_name, **kwargs)
        if article_prompt is None:
            article_prompt = self.default_article_prompt()
        if task_prompt is None:
            task_prompt = self.default_task_prompt()
        self.system_prompt = system_prompt
        self.article_prompt = article_prompt
        self.task_prompt = task_prompt

    def default_task_prompt(self):
        return "TL;DR:"

    def default_article_prompt(self):
        return "Article: {article}"

    def truncate_input(self, prompt, max_tokens, **kwargs):
        # discount maximum output length from max_tokens
        max_length = kwargs.get("max_length", 256)
        max_tokens = max_tokens - max_length - 1
        num_tokens = self.num_tokens_for_prompt(prompt)
        excess_tokens = num_tokens - max_tokens

        # choose the longest message for truncation
        message_lengths = [len(m["content"]) for m in prompt]
        longest_idx = np.argmax(message_lengths)

        if excess_tokens > 0:
            tokenizer_kwargs = self.get_tokenizer_kwargs()
            tokenizer = self.load_tokenizer(self.model_name, **tokenizer_kwargs)
            truncated_prompt = [dict(item) for item in prompt]
            text = prompt[longest_idx]["content"]
            tokens = tokenizer.encode(text)
            tokens = tokens[:-excess_tokens]
            try:
                truncated_prompt[longest_idx]["content"] = tokenizer.decode(tokens, 
                                                                            skip_special_tokens=True, 
                                                                            clean_up_tokenization_spaces=None)
            except TypeError:
                truncated_prompt[longest_idx]["content"] = tokenizer.decode(tokens)

            prompt = truncated_prompt
        return prompt, excess_tokens

    def build_input(
        self,
        text,
        system_prompt=None,
        article_prompt=None,
        task_prompt=None,
        num_sentences=6,
        verbose=False,
        **generation_kwargs,
    ):
        if article_prompt is None:
            article_prompt = self.article_prompt
        if task_prompt is None:
            task_prompt = self.task_prompt
        if system_prompt is None:
            system_prompt = self.system_prompt

        prompt_args = dict(
            article=text,
            num_sentences=num_sentences, 
            **generation_kwargs)
        article_prompt = article_prompt.format(**prompt_args)
        prompt = []

        if system_prompt:
            prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": article_prompt})

        if task_prompt:
            task_prompt = task_prompt.format(**prompt_args)
            prompt.append({"role": "user", "content": task_prompt})

        log(logger, f"System prompt: {pformat(system_prompt)}", verbose=verbose)
        log(logger, f"Article prompt: {pformat(article_prompt)}", verbose=verbose)
        log(logger, f"Task prompt: {pformat(task_prompt)}", verbose=verbose)
        return prompt, generation_kwargs

    def num_tokens_for_prompt(self, messages):
        tokenizer_kwargs = self.get_tokenizer_kwargs()
        tokenizer = self.load_tokenizer(self.model_name, **tokenizer_kwargs)
        # account for one newline after each message
        newline_tokens = tokenizer.encode("\n")
        num_tokens = len(newline_tokens) * (len(messages) - 1)
        for message in messages:
            num_tokens += len(tokenizer.encode(message["content"]))
        return num_tokens

    def token_statistics_for_input(self, text, truncation, **generation_kwargs):
        prompt, truncated_tokens, _ = self.preprocess(text, 
                                                      truncation=truncation, 
                                                      **generation_kwargs)
        tokenizer_kwargs = self.get_tokenizer_kwargs()
        tokenizer = self.load_tokenizer(self.model_name, **tokenizer_kwargs)
        if not isinstance(prompt, str):
            prompt = self.prompt_to_text(prompt)
        num_tokens = len(tokenizer.encode(prompt))
        return num_tokens, truncated_tokens

    def token_statistics(
        self,
        texts,
        truncation=True,
        **generation_kwargs,
    ):
        progress = get_progress_bar()
        task = add_progress_task(
            progress,
            f"Calculating token statistics for {self.model_name}...",
            total=len(texts),
            existing_ok=False,
        )
        truncated_tokens = []
        num_tokens = []
        
        with progress:
            for text in texts:
                result = self.token_statistics_for_input(text, truncation, 
                                                         **generation_kwargs)
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


class InstructTunedSummarizer(PromptBasedSummarizer):
    def __init__(self, model_name_or_path, **kwargs) -> None:
        super().__init__(model_name_or_path, **kwargs)

    def default_task_prompt(self):
        return "Summarize the article above in {num_sentences} sentences."
