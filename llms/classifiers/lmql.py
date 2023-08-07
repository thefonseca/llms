import logging

import lmql
import torch

from ..models.huggingface import HFModel, CausalLM, Vicuna, LlamaChat
from .models import InstructTunedClassifier
from ..utils.memoizer import memoize

logger = logging.getLogger(__name__)


class LMQLInstructClassifier(InstructTunedClassifier, HFModel):
    def __init__(self, model_name, labels, **kwargs) -> None:
        self.lmql_model_name = model_name.replace("lmql:", "")

        if "checkpoint_path" in kwargs:
            self.lmql_model_path = kwargs["checkpoint_path"].replace("lmql:", "")
            kwargs["checkpoint_path"] = kwargs["checkpoint_path"].replace("local:", "")
        else:
            self.lmql_model_path = self.lmql_model_name

        model_name = self.lmql_model_name.replace("local:", "")
        if "local:" in self.lmql_model_name or "local:" in self.lmql_model_path:
            torch.multiprocessing.set_start_method("spawn")
        super().__init__(model_name, labels, **kwargs)
        label_words = [len(l.split(" ")) for l in labels]
        self.max_label_words = max(label_words) + 1

    def load_model(self, **kwargs):
        if self.model:
            return self.model

        logger.info(f"Loading model {self.model_path}...")

        if "local:" in self.lmql_model_path:
            model = lmql.model(self.lmql_model_path, **kwargs)
        else:
            model = self.lmql_model_path
        self.model = model
        return model

    def default_user_prompt(self):
        return (
            "Text: {input}\n\nClassify the text above into one of the following categories:\n{labels}\n"
            "Be concise and only write the category name."
        )

    def process_generation_kwargs(self, **generation_kwargs):
        kwargs = super().process_generation_kwargs(**generation_kwargs)
        if "lmql" not in kwargs:
            kwargs["lmql"] = '"{query} Category: [CATEGORY]" where CATEGORY in labels'
        return kwargs

    def prompt_to_text(self, prompt):
        prompt = super().prompt_to_text(prompt)
        prompt = prompt.replace("\n", "\\n")
        return prompt

    @memoize()
    def generate_cached(
        self,
        model_name,
        model_input,
        decoder="argmax",
        n_samples=1,
        temperature=0,
        **generation_kwargs,
    ):
        model_kwargs = self.get_model_kwargs()
        model = self.load_model(**model_kwargs)
        lmql_str = generation_kwargs["lmql"]
        model_input = model_input.replace('"', '\\"')
        lmql_str = lmql_str.format(query=model_input)

        result = lmql.run_sync(
            lmql_str,
            labels=self.labels,
            model=model,
            decoder=decoder,
            temperature=temperature,
            n=n_samples,
            **generation_kwargs,
        )
        result = result[0].variables["CATEGORY"].strip()
        return result


class LMQLInstructCausalLMClassifier(LMQLInstructClassifier, CausalLM):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class LMQLVicunaClassifier(Vicuna, LMQLInstructCausalLMClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class LMQLLlamaChatClassifier(LlamaChat, LMQLInstructCausalLMClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def default_system_prompt(self):
        return None

    def prompt_to_text(self, prompt):
        prompt = super().prompt_to_text(prompt)
        prompt = prompt.replace("[INST]", "[[INST]]")
        prompt = prompt.replace("[/INST]", "[[/INST]]")
        prompt = prompt.replace("\n", "\\n")
        return prompt
