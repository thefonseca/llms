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
        dtype="auto",
        **kwargs,
    ):
        super().__init__(model_name_or_path, **kwargs)
        self.cache_dir = cache_dir
        self.device_map = device_map
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.load_in_8bit = load_in_8bit
        self.dtype = dtype
        self.tokenizer = None
        self.model = None
        self.model_config = None
        self.generation_config = None

    @property
    def dtype(self):
        return self._dtype
    
    @dtype.setter
    def dtype(self, value):
        if value == "fp16":
            dtype = torch.float16
        elif value == "fp32":
            dtype = torch.float32
        elif value == "fp64":
            dtype = torch.float64
        elif value == "bf16":
            dtype = torch.bfloat16
        elif value == "auto":
            dtype = "auto"
        else:
            logger.warning(f"Unsupported dtype {value}. Setting dtype = 'auto'.")
            dtype = "auto"
        self._dtype = dtype
        

    def get_model_kwargs(self):
        return dict(
            cache_dir=self.cache_dir,
            device_map=self.device_map,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            load_in_8bit=self.load_in_8bit,
            dtype=self.dtype,
        )

    def get_tokenizer_kwargs(self):
        return dict(cache_dir=self.cache_dir)

    def load_tokenizer(self, model_name_or_path, **kwargs):
        if self.tokenizer:
            return self.tokenizer

        logger.info(f"Loading tokenizer {model_name_or_path}...")
        if "google/pegasus-x-base" in model_name_or_path:
            model_name_or_path = "google/pegasus-x-base"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, **kwargs
        )
        self.tokenizer = tokenizer
        return tokenizer

    def load_model_config(self, model_name_or_path, **kwargs):
        if self.model_config is not None:
            return self.model_config

        try:
            model_config = AutoConfig.from_pretrained(
                model_name_or_path, **kwargs
            )
            self.model_config = model_config
        except OSError:
            logger.warning(
                f"{model_name_or_path} does not appear to have a file named config.json"
            )
            self.model_config = AutoConfig()
        return self.model_config
    
    def load_generation_config(self, model_name_or_path, **kwargs):
        if self.generation_config is not None:
            return self.generation_config

        try:
            generation_config = GenerationConfig.from_pretrained(
                model_name_or_path, **kwargs
            )
            self.generation_config = generation_config
        except OSError:
            logger.warning(
                f"{model_name_or_path} does not appear to have a file named generation_config.json"
            )
            self.generation_config = GenerationConfig()
        return self.generation_config

    def default_max_tokens(self):
        config = self.load_model_config(self.model_name, cache_dir=self.cache_dir)
        if "google/bigbird-pegasus-large" in self.model_name:
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
        dtype="auto",
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
            dtype=dtype,
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
        dtype="auto",
        **kwargs,
    ):
        if self.model:
            return self.model

        logger.info(f"Loading model {model_name_or_path}...")
        if load_in_8bit:
            dtype = torch.int8
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

        config = self.load_model_config(self.model_name, cache_dir=self.cache_dir)
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
        dtype="auto",
        **kwargs,
    ):
        if self.model:
            return self.model

        logger.info(f"Loading model {model_name_or_path}...")
        if load_in_8bit:
            dtype = torch.int8
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
            max_length = self.default_max_tokens()
            max_new_tokens = max_length - num_input_tokens
        else:
            max_new_tokens = max_length
        generation_kwargs["max_new_tokens"] = max_new_tokens
        generation_kwargs["keep_generated_only"] = True
        prompt_text = self.prompt_to_text(prompt)
        return prompt_text, generation_kwargs


class InstructCausalLMSummarizer(InstructTunedSummarizer, CausalLMSummarizer):
    def __init__(self, model_name_or_path, **kwargs) -> None:
        super().__init__(model_name_or_path, **kwargs)


class InstructText2TextSummarizer(InstructTunedSummarizer, Text2TextSummarizer):
    def __init__(self, model_name_or_path, **kwargs) -> None:
        super().__init__(model_name_or_path, **kwargs)

    def truncate_input(self, prompt, max_tokens, **kwargs):
        prompt = super().truncate_input(prompt, max_tokens, **kwargs)
        prompt_text = self.prompt_to_text(prompt)
        return prompt_text
    

class T5Summarizer(InstructText2TextSummarizer):
    def __init__(self, model_name_or_path, **kwargs) -> None:
        super().__init__(model_name_or_path, **kwargs)

    def default_article_prompt(self):
        # From promptsource CNN/DM template:
        # https://github.com/bigscience-workshop/promptsource
        return "Summarize the article: {}"
    
    def default_task_prompt(self):
        return None


class AlpacaSummarizer(InstructCausalLMSummarizer):
    def __init__(self, model_name_or_path, **kwargs) -> None:
        super().__init__(model_name_or_path, **kwargs)

    def prompt_to_text(self, prompt):
        prompt_text = super().prompt_to_text(prompt)
        prompt_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{}\n\n### Response:"
        ).format(prompt_text)
        return prompt_text
