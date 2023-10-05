import logging
import re

from accelerate import infer_auto_device_map, init_empty_weights
from fastchat.conversation import get_conv_template
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
)

from ..models.base import BaseLM, PromptBasedLM
from ..utils.memoizer import memoize

logger = logging.getLogger(__name__)


class HFModel(BaseLM):
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
        if value is None or not isinstance(value, str):
            dtype = value
        elif value == "auto":
            dtype = "auto"
        elif hasattr(torch, value):
            dtype = getattr(torch, value)
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
        self,
        model_path,
        device_map="auto",
        max_memory=None,
        dtype=None,
        tie_weights=False,
    ):
        # Example: max_memory = {0: "22GiB", "cpu": "30GiB"}
        if max_memory:
            logger.info(f"Inferring device map for {max_memory}...")
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(model_path)
            if tie_weights:
                model.tie_weights()
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

    def process_generation_kwargs(self, max_length=None, **generation_kwargs):
        if "max_new_tokens" not in generation_kwargs:
            if max_length is None:
                generation_config = self.load_generation_config()
                if hasattr(generation_config, "max_length"):
                    max_length = generation_config.max_length

            generation_kwargs["max_new_tokens"] = max_length

        generation_kwargs.pop("max_tokens", None)
        return generation_kwargs

    def preprocess(self, text, truncation=True, **generation_kwargs):
        prompt, truncated_tokens, generation_kwargs = super().preprocess(
            text, truncation=truncation, **generation_kwargs
        )
        generation_kwargs = self.process_generation_kwargs(**generation_kwargs)
        return prompt, truncated_tokens, generation_kwargs

    @memoize(ignore_kwargs=["model_path", "max_memory"])
    def generate_cached(
        self,
        model_name,
        model_input,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        seed=42,
        keep_generated_only=False,
        input_length_adjust=0,
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
            output = tokenizer.batch_decode(
                generated_ids.cpu(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=None,
            )[0]
            if isinstance(model_input, str) and keep_generated_only:
                output = output[len(model_input) + input_length_adjust :]
            return output


class Text2TextLM(HFModel):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def load_model(
        self,
        device_map="auto",
        max_memory=None,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        tie_weights=False,
        dtype="auto",
        **kwargs,
    ):
        if self.model:
            return self.model

        logger.info(f"Loading model {self.model_path}...")
        if any(
            [x in self.model_path] for x in ["google/pegasus", "google/bigbird-pegasus"]
        ):
            device_map = None
            low_cpu_mem_usage = False
        else:
            device_map = self.infer_device_map(
                self.model_path,
                max_memory=max_memory,
                tie_weights=tie_weights,
                dtype=dtype,
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

    def process_generation_kwargs(self, do_sample=None, **generation_kwargs):
        generation_kwargs = super().process_generation_kwargs(**generation_kwargs)
        if do_sample is None:
            generation_kwargs["do_sample"] = False
        return generation_kwargs

    def postprocess(self, output):
        # special newline postprocessing for some models
        if any(
            [x in self.model_path] for x in ["google/pegasus", "google/bigbird-pegasus"]
        ):
            output = output.replace(".<n> ", ".\n ")
        output = super().postprocess(output)
        return output


class CausalLM(HFModel):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def load_model(
        self,
        device_map="auto",
        max_memory=None,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        tie_weights=False,
        dtype="auto",
        **kwargs,
    ):
        if self.model:
            return self.model

        logger.info(f"Loading model {self.model_path}...")
        device_map = self.infer_device_map(
            self.model_path,
            device_map=device_map,
            max_memory=max_memory,
            tie_weights=tie_weights,
            dtype=dtype,
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

    def process_generation_kwargs(self, **generation_kwargs):
        generation_kwargs = super().process_generation_kwargs(**generation_kwargs)
        generation_kwargs["keep_generated_only"] = True
        return generation_kwargs


class InstructText2TextLM(PromptBasedLM, Text2TextLM):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def truncate_input(self, prompt, max_tokens, **kwargs):
        prompt, truncated_tokens = super().truncate_input(prompt, max_tokens, **kwargs)
        prompt_text = self.prompt_to_text(prompt)
        return prompt_text, truncated_tokens


class InstructCausalLM(PromptBasedLM, CausalLM):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def preprocess(self, text, truncation=True, max_length=None, **generation_kwargs):
        prompt, truncated_tokens, generation_kwargs = super().preprocess(
            text, truncation=truncation, max_length=max_length, **generation_kwargs
        )
        if max_length is None:
            num_input_tokens = self.num_tokens_for_prompt(prompt)
            max_length = self.default_max_tokens()
            max_new_tokens = max_length - num_input_tokens
        else:
            max_new_tokens = max_length
        generation_kwargs["max_new_tokens"] = max_new_tokens
        prompt_text = self.prompt_to_text(prompt)
        return prompt_text, truncated_tokens, generation_kwargs

    def postprocess(self, output):
        # special newline postprocessing for some models
        if "mosaicml/mpt-7b" in self.model_path:
            output = [s for s in re.split("[\n\s#]+$", output) if len(s)]
            output = [re.sub("^[\n\s#]+", "", s.strip()).strip() for s in output]
            output = "\n".join(output)
        output = super().postprocess(output)
        return output


class Alpaca(InstructCausalLM):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def default_system_prompt(self):
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
        )

    def prompt_to_text(self, prompt):
        system_prompt = [m for m in prompt if m["role"] == "system"]
        user_message = [m for m in prompt if m["role"] in ["input", "user"]]
        prompt_text = "\n".join([m["content"] for m in user_message])

        if system_prompt:
            system_prompt = system_prompt[0]["content"]
            prompt_text = (
                f"{system_prompt}### Instruction:\n{user_message}\n\n### Response:"
            )
        return prompt_text


class Vicuna(InstructCausalLM):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def prompt_to_text(self, prompt):
        prompt_text = super().prompt_to_text(prompt)
        conv = get_conv_template("vicuna_v1.1").copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        return prompt_text


class LlamaChat(InstructCausalLM):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def default_max_tokens(self):
        return 4096

    def load_tokenizer(self):
        tokenizer = super().load_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def default_system_prompt(self):
        return "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."

    def prompt_to_text(self, prompt):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        system_prompt = [m for m in prompt if m["role"] == "system"]
        user_message = [m for m in prompt if m["role"] in ["input", "user"]]
        user_message = "\n".join([m["content"] for m in user_message])

        if system_prompt:
            system_prompt = system_prompt[0]["content"]
            prompt_text = (
                f"{B_INST} {B_SYS}{system_prompt}{E_SYS}{user_message} {E_INST}"
            )
        else:
            prompt_text = f"{B_INST} {user_message} {E_INST}"
        return prompt_text

    def process_generation_kwargs(self, **generation_kwargs):
        generation_kwargs = super().process_generation_kwargs(**generation_kwargs)
        # account for the added '<s>' token in the prompt
        generation_kwargs["input_length_adjust"] = -1
        if "temperature" not in generation_kwargs:
            generation_kwargs["temperature"] = 0.8
        return generation_kwargs


class FalconChat(InstructCausalLM):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def default_max_tokens(self):
        return 2048

    def load_tokenizer(self):
        tokenizer = super().load_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def process_generation_kwargs(self, **generation_kwargs):
        generation_kwargs = super().process_generation_kwargs(**generation_kwargs)
        generation_kwargs["pad_token_id"] = self.load_tokenizer().eos_token_id
        return generation_kwargs

    def prompt_to_text(self, prompt):
        system_prompt = [m for m in prompt if m["role"] == "system"]
        user_message = [m for m in prompt if m["role"] in ["input", "user"]]
        user_message = "\n".join([m["content"] for m in user_message])

        if system_prompt:
            system_prompt = system_prompt[0]["content"]
            prompt_text = f"System: {system_prompt}\nUser: {user_message}\nFalcon:"
        else:
            prompt_text = f"User: {user_message}\nFalcon:"
        return prompt_text


class MistralInstruct(LlamaChat):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def default_system_prompt(self):
        return None
    
    def process_generation_kwargs(self, **generation_kwargs):
        generation_kwargs = super().process_generation_kwargs(**generation_kwargs)
        generation_kwargs["input_length_adjust"] = -2
        generation_kwargs["pad_token_id"] = self.load_tokenizer().eos_token_id
        return generation_kwargs

    def prompt_to_text(self, prompt):
        prompt_text = super().prompt_to_text(prompt)
        prompt_text = f"<s>{prompt_text}"
        return prompt_text
