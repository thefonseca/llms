import logging

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
)

from .base import Summarizer, PromptBasedSummarizer, InstructTunedSummarizer
from ..memoizer import memoize

logger = logging.getLogger(__name__)


class HFSummarizer(Summarizer):
    def __init__(
        self,
        model_name_or_path,
        cache_dir=None,
        device_map="auto",
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        **kwargs,
    ):
        super().__init__(model_name_or_path, **kwargs)
        self.cache_dir = cache_dir
        self.device_map = device_map
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.load_in_8bit = load_in_8bit
        self.tokenizer = None
        self.model = None
        self.generation_config = None

    def get_model_kwargs(self):
        return dict(
            cache_dir=self.cache_dir,
            device_map=self.device_map,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            load_in_8bit=self.load_in_8bit,
        )

    def get_tokenizer_kwargs(self):
        return dict(cache_dir=self.cache_dir)

    def load_tokenizer(self, model_name_or_path, cache_dir=None, **kwargs):
        if self.tokenizer:
            return self.tokenizer

        logger.info(f"Loading tokenizer {model_name_or_path}...")
        if "google/pegasus-x-bgeneration_conase" in model_name_or_path:
            model_name_or_path = "google/pegasus-x-base"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, **kwargs
        )
        self.tokenizer = tokenizer
        return tokenizer

    def load_generation_config(self, model_name_or_path, cache_dir=None, **kwargs):
        if self.generation_config is not None:
            return self.generation_config

        try:
            generation_config = GenerationConfig.from_pretrained(
                model_name_or_path, cache_dir, **kwargs
            )
            self.generation_config = generation_config
        except OSError:
            logger.warning(
                f"{model_name_or_path} does not appear to have a file named generation_config.json"
            )
            self.generation_config = GenerationConfig()
        return self.generation_config

    @staticmethod
    def default_max_tokens(model_name):
        config = AutoConfig.from_pretrained(model_name)
        if "google/bigbird-pegasus-large" in model_name:
            max_tokens = 3072
        elif hasattr(config, "max_position_embeddings"):
            max_tokens = config.max_position_embeddings
        elif hasattr(config, "max_encoder_position_embeddings"):
            max_tokens = config.max_encoder_position_embeddings
        elif hasattr(config, "max_sequence_length"):
            max_tokens = config.max_sequence_length
        elif hasattr(config, "n_positions"):
            max_tokens = config.n_positions
        else:
            max_tokens = 1024
        return max_tokens

    @memoize()
    def generate_cached(
        self,
        model_name_or_path,
        model_input,
        cache_dir=None,
        device_map="auto",
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        keep_generated_only=False,
        memoizer_ignore_cache=False,
        **generation_kwargs,
    ):
        model = self.load_model(
            model_name_or_path,
            cache_dir=cache_dir,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
            load_in_8bit=load_in_8bit,
        )
        tokenizer = self.load_tokenizer(model_name_or_path, cache_dir=cache_dir)
        generation_config = self.load_generation_config(
            model_name_or_path, cache_dir=cache_dir
        )

        with torch.no_grad():
            if isinstance(model_input, str):
                input_ids = tokenizer(model_input, return_tensors="pt").input_ids
            else:
                input_ids = model_input.input_ids

            if isinstance(model, LlamaForCausalLM):
                # If the prompt is ending with EOS, often the generation will stop abruptly.
                if input_ids[0][-1] == tokenizer.eos_token_id:
                    input_ids = input_ids[:, :-1]

            input_ids = input_ids.to(0)
            generated_ids = model.generate(
                input_ids,
                generation_config,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **generation_kwargs,
            )
            summary = tokenizer.batch_decode(
                generated_ids.cpu(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            if isinstance(model_input, str) and keep_generated_only:
                summary = summary[len(model_input) :]
            return summary


class Text2TextSummarizer(HFSummarizer):
    def __init__(self, model_name_or_path, **kwargs) -> None:
        super().__init__(model_name_or_path, **kwargs)

    def load_model(
        self,
        model_name_or_path,
        cache_dir=None,
        device_map="auto",
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        **kwargs,
    ):
        if self.model:
            return self.model

        logger.info(f"Loading model {model_name_or_path}...")
        dtype = torch.int8 if load_in_8bit else "auto"
        if any(
            [x in self.model_name] for x in ["google/pegasus", "google/bigbird-pegasus"]
        ):
            device_map = None
            low_cpu_mem_usage = False

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
            load_in_8bit=load_in_8bit,
            torch_dtype=dtype,
            **kwargs,
        )
        if device_map is None:
            try:
                device = torch.cuda.current_device()
                model = model.to(device)
            except:
                logger.warning("Failed to get cuda device")

        tokenizer = self.load_tokenizer(model_name_or_path, cache_dir=cache_dir)
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        if model.config.decoder_start_token_id is None:
            raise ValueError(
                "Make sure that `config.decoder_start_token_id` is correctly defined"
            )
        model.eval()
        self.model = model
        return model

    def truncate_input(self, input, max_tokens, **kwargs):
        tokenizer = self.load_tokenizer(self.model_name, cache_dir=self.cache_dir)
        input = tokenizer(
            [input],
            padding="max_length",
            max_length=max_tokens,
            truncation=True,
            return_tensors="pt",
        )
        return input

    def preprocess(
        self,
        text,
        truncation=True,
        max_length=None,
        do_sample=None,
        **generation_kwargs,
    ):
        model_input, generation_kwargs = super().preprocess(
            text, truncation=truncation, **generation_kwargs
        )
        if max_length is None:
            generation_config = self.load_generation_config(
                self.model_name, self.cache_dir
            )
            if hasattr("max_length", generation_config):
                max_length = generation_config.max_length
        generation_kwargs["max_new_tokens"] = max_length
        if do_sample is None:
            generation_kwargs["do_sample"] = False

        config = AutoConfig.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        if hasattr(config, "task_specific_params"):
            task_params = config.task_specific_params
            if task_params and "summarization" in task_params:
                for key, param in task_params["summarization"].items():
                    if (
                        key not in ["prefix", "max_length"]
                        and key not in generation_kwargs
                    ):
                        generation_kwargs[key] = param

        return model_input, generation_kwargs

    def postprocess(self, summary):
        # special newline postprocessing for some models
        if any(
            [x in self.model_name] for x in ["google/pegasus", "google/bigbird-pegasus"]
        ):
            summary = summary.replace(".<n> ", ".\n ")
        summary = super().postprocess(summary)
        return summary


class CausalLMSummarizer(PromptBasedSummarizer, HFSummarizer):
    def __init__(self, model_name_or_path, **kwargs) -> None:
        super().__init__(model_name_or_path, **kwargs)

    def load_model(
        self,
        model_name_or_path,
        cache_dir=None,
        device_map="auto",
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        **kwargs,
    ):
        if self.model:
            return self.model

        logger.info(f"Loading model {model_name_or_path}...")
        dtype = torch.int8 if load_in_8bit else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
            load_in_8bit=load_in_8bit,
            torch_dtype=dtype,
            **kwargs,
        )
        self.model = model
        return model

    def preprocess(self, text, truncation=True, max_length=None, **generation_kwargs):
        prompt, generation_kwargs = super().preprocess(
            text, truncation=truncation, **generation_kwargs
        )
        if max_length is None:
            num_input_tokens = self.num_tokens_for_prompt(prompt)
            max_length = self.default_max_tokens(self.model_name)
            max_new_tokens = max_length - num_input_tokens
        else:
            max_new_tokens = max_length
        generation_kwargs["max_new_tokens"] = max_new_tokens
        generation_kwargs["keep_generated_only"] = True
        prompt_text = "\n".join([m["content"] for m in prompt])
        return prompt_text, generation_kwargs


class T5Summarizer(InstructTunedSummarizer, Text2TextSummarizer):
    def __init__(self, model_name_or_path, **kwargs) -> None:
        super().__init__(model_name_or_path, **kwargs)

    def truncate_input(self, prompt, max_tokens, **kwargs):
        prompt = super().truncate_input(prompt, max_tokens, **kwargs)
        prompt_text = "\n".join([m["content"] for m in prompt])
        return prompt_text
