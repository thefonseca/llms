import logging

from accelerate import infer_auto_device_map, init_empty_weights
from fastchat.conversation import get_default_conv_template
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
        model_name,
        checkpoint_path=None,
        **kwargs,
    ):
        super().__init__(model_name)
        self.tokenizer = None
        self.model = None
        self.model_config = None
        self.generation_config = None
        self.model_path = checkpoint_path
        if self.model_path is None:
            self.model_path = model_name
        self.model_kwargs = kwargs
        if "dtype" in self.model_kwargs:
            dtype = self.model_kwargs["dtype"]
            self.model_kwargs["dtype"] = self.infer_dtype(dtype)
        
    def infer_dtype(self, value):
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
        return dtype

    def get_model_kwargs(self):
        return dict(**self.model_kwargs)

    def get_tokenizer_kwargs(self):
        cache_dir = self.model_kwargs.get("cache_dir")
        return dict(cache_dir=cache_dir)

    def get_generation_config_kwargs(self):
        cache_dir = self.model_kwargs.get("cache_dir")
        return dict(cache_dir=cache_dir)

    def load_tokenizer(self):
        if self.tokenizer:
            return self.tokenizer

        model_path = self.model_path
        logger.info(f"Loading tokenizer {model_path}...")
        if "google/pegasus-x-base" in model_path:
            model_path = "google/pegasus-x-base"

        kwargs = self.get_tokenizer_kwargs()
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        self.tokenizer = tokenizer
        return tokenizer

    def load_model_config(self):
        if self.model_config is not None:
            return self.model_config

        try:
            kwargs = self.get_model_kwargs()
            model_config = AutoConfig.from_pretrained(self.model_path, **kwargs)
            self.model_config = model_config
        except OSError:
            logger.warning(
                f"{self.model_path} does not appear to have a file named config.json"
            )
            self.model_config = AutoConfig()
        return self.model_config

    def load_generation_config(self):
        if self.generation_config is not None:
            return self.generation_config

        try:
            kwargs = self.get_generation_config_kwargs()
            generation_config = GenerationConfig.from_pretrained(
                self.model_path, **kwargs
            )
            self.generation_config = generation_config
        except OSError:
            logger.warning(
                f"{self.model_path} does not appear to have a file named generation_config.json"
            )
            self.generation_config = GenerationConfig()
        return self.generation_config

    def default_max_tokens(self):
        config = self.load_model_config()
        if "google/bigbird-pegasus-large" in self.model_path:
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

    def infer_device_map(
        self, model_path, device_map="auto", max_memory=None, dtype=None
    ):
        # Example: max_memory = {0: "22GiB", "cpu": "30GiB"}
        if max_memory:
            logger.info(f"Inferring device map for {max_memory}...")
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(model_path)
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[
                    "BloomBlock",
                    "OPTDecoderLayer",
                    "LLaMADecoderLayer",
                    "LlamaDecoderLayer",
                ],
                dtype=dtype,
            )
        logger.info(f"Using device map: {device_map}")
        return device_map

    @memoize(ignore_kwargs=["model_path"])
    def generate_cached(
        self,
        model_name,
        model_input,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        seed=42,
        keep_generated_only=False,
        memoizer_ignore_cache=False,
        **kwargs,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        logger.debug(f"Performing generation for model {model_name}...")
        model_kwargs = self.get_model_kwargs()
        generation_kwargs = {}
        for k, v in kwargs.items():
            if k not in model_kwargs:
                generation_kwargs[k] = v

        model = self.load_model(**model_kwargs)
        tokenizer = self.load_tokenizer()
        generation_config = self.load_generation_config()

        with torch.no_grad():
            if isinstance(model_input, str):
                input_ids = tokenizer(model_input, return_tensors="pt").input_ids
            else:
                input_ids = model_input.input_ids

            if isinstance(model, LlamaForCausalLM):
                # If the prompt is ending with EOS, often the generation will stop abruptly.
                EOS_TOKEN_ID = 2
                if input_ids[0][-1] == EOS_TOKEN_ID:
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
                clean_up_tokenization_spaces=None,
            )[0]
            if isinstance(model_input, str) and keep_generated_only:
                summary = summary[len(model_input) :]
            return summary


class Text2TextSummarizer(HFSummarizer):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def load_model(
        self,
        device_map="auto",
        max_memory=None,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        dtype="auto",
        **kwargs,
    ):
        if self.model:
            return self.model

        logger.info(f"Loading model {self.model_path}...")
        if load_in_8bit:
            dtype = torch.int8
        
        if any(
            [x in self.model_path] for x in ["google/pegasus", "google/bigbird-pegasus"]
        ):
            device_map = None
            low_cpu_mem_usage = False
        else:
            device_map = self.infer_device_map(
                self.model_path, max_memory=max_memory, dtype=dtype
            )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
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

        tokenizer = self.load_tokenizer()
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
        tokenizer = self.load_tokenizer()
        input = tokenizer(
            [input],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        n_tokens = len(input["input_ids"])
        truncated_tokens = 0

        if n_tokens > max_tokens:
            input = tokenizer(
                [input],
                padding="max_length",
                max_length=max_tokens,
                truncation=True,
                return_tensors="pt",
            )
            truncated_tokens = n_tokens - max_tokens

        return input, truncated_tokens

    def preprocess(
        self,
        text,
        truncation=True,
        max_length=None,
        do_sample=None,
        **generation_kwargs,
    ):
        model_input, truncated_tokens, generation_kwargs = super().preprocess(
            text, truncation=truncation, **generation_kwargs
        )
        if max_length is None:
            generation_config = self.load_generation_config()
            if hasattr("max_length", generation_config):
                max_length = generation_config.max_length

        generation_kwargs["max_new_tokens"] = max_length
        if do_sample is None:
            generation_kwargs["do_sample"] = False

        config = self.load_model_config()
        if hasattr(config, "task_specific_params"):
            task_params = config.task_specific_params
            if task_params and "summarization" in task_params:
                for key, param in task_params["summarization"].items():
                    if (
                        key not in ["prefix", "max_length"]
                        and key not in generation_kwargs
                    ):
                        generation_kwargs[key] = param

        return model_input, truncated_tokens, generation_kwargs

    def postprocess(self, summary):
        # special newline postprocessing for some models
        if any(
            [x in self.model_path] for x in ["google/pegasus", "google/bigbird-pegasus"]
        ):
            summary = summary.replace(".<n> ", ".\n ")
        summary = super().postprocess(summary)
        return summary


class CausalLMSummarizer(PromptBasedSummarizer, HFSummarizer):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def load_model(
        self,
        device_map="auto",
        max_memory=None,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        dtype="auto",
        **kwargs,
    ):
        if self.model:
            return self.model

        logger.info(f"Loading model {self.model_path}...")
        if load_in_8bit:
            dtype = torch.int8
        
        device_map = self.infer_device_map(
            self.model_path, device_map=device_map, max_memory=max_memory, dtype=dtype
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
            load_in_8bit=load_in_8bit,
            torch_dtype=dtype,
            **kwargs,
        )
        self.model = model
        return model

    def preprocess(self, text, truncation=True, max_length=None, **generation_kwargs):
        prompt, truncated_tokens, generation_kwargs = super().preprocess(
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
        return prompt_text, truncated_tokens, generation_kwargs


class InstructCausalLMSummarizer(InstructTunedSummarizer, CausalLMSummarizer):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class InstructText2TextSummarizer(InstructTunedSummarizer, Text2TextSummarizer):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def truncate_input(self, prompt, max_tokens, **kwargs):
        prompt, truncated_tokens = super().truncate_input(prompt, max_tokens, **kwargs)
        prompt_text = self.prompt_to_text(prompt)
        return prompt_text, truncated_tokens


class T5Summarizer(InstructText2TextSummarizer):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def default_article_prompt(self):
        return "summarize: {article}"

    def default_task_prompt(self):
        return None


class AlpacaSummarizer(InstructCausalLMSummarizer):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def prompt_to_text(self, prompt):
        prompt_text = super().prompt_to_text(prompt)
        prompt_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{}\n\n### Response:"
        ).format(prompt_text)
        return prompt_text


class VicunaSummarizer(InstructCausalLMSummarizer):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def prompt_to_text(self, prompt):
        prompt_text = super().prompt_to_text(prompt)
        conv = get_default_conv_template("vicuna").copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        return prompt_text
