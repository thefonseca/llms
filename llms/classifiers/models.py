import logging
import random
import re

import numpy as np
import torch
import torch.nn.functional as F

from ..models.base import BaseLM
from ..models.huggingface import (
    HFModel,
    InstructCausalLM,
    Alpaca,
    Vicuna,
    LlamaChat,
)
from ..models.openai import OpenAIChat
from ..utils.memoizer import memoize

logger = logging.getLogger(__name__)


class BaseClassifier(BaseLM):
    def __init__(self, model_name, labels, label_type="category", **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        self.labels = labels
        self.label_type = label_type

    def get_prompt_args(self):
        label_type = self.label_type.capitalize()
        return dict(label_type=label_type)

    def process_generation_kwargs(self, **generation_kwargs):
        kwargs = super().process_generation_kwargs(**generation_kwargs)
        if "labels" not in kwargs:
            kwargs["labels"] = self.labels
        if "label_type" not in kwargs:
            kwargs["label_type"] = self.label_type
        return kwargs

    def postprocess(self, output):
        output = super().postprocess(output)
        # Labels can be specified as a map from a language label to a symbol.
        # E.g., {"Positive":1, "Negative": 0}
        if isinstance(self.labels, dict):
            output = self.labels.get(output, output)
        return output
    

class HFClassifier(BaseClassifier, HFModel):
    def __init__(self, model_name, labels, **kwargs) -> None:
        super().__init__(model_name, labels, **kwargs)

    def process_generation_kwargs(self, **generation_kwargs):
        kwargs = super().process_generation_kwargs(**generation_kwargs)
        if "do_sample" not in kwargs:
            kwargs["do_sample"] = False
        return kwargs


class InstructTunedClassifier(BaseClassifier):
    def __init__(self, model_name, labels, **kwargs) -> None:
        super().__init__(model_name, labels, **kwargs)

    def get_prompt_args(self):
        args = super().get_prompt_args()
        labels = self.labels
        if isinstance(labels, (list, tuple, dict)):
            labels = [f"- {label}" for label in labels]
            labels = "\n".join(labels)
        args["labels"] = labels
        return args

    def default_input_prompt(self):
        return "Text: {input}"
            
    def default_user_prompt(self):
        return (
            "\nClassify the text above into one of the following categories:\n{labels}\n"
            "Be concise and only write the category name."
            "\n{label_type}:"
        )

    def fix_prediction(self, output):
        if isinstance(self.labels, (list, tuple, dict)):
            target_labels = [x.lower() for x in self.labels]
        else:
            target_labels = self.labels.lower()

        if output == "":
            logger.warning(f"Prediction is empty")

        elif output.lower() not in target_labels:
            output = re.sub(r"(C|c)ategory:\s*", "", output)
            output = output.strip()

            if output.lower() not in target_labels:
                for label in self.labels:
                    # some models output truncated labels
                    if output.lower() == label.lower()[: len(output)]:
                        logger.warning(f"Fixing prediction: {output} => {label}")
                        output = label
                        break
                else:
                    logger.warning(
                        f'Prediction "{output}" is not in labels: {self.labels}.'
                    )

        return output

    def postprocess(self, output):
        output = self.fix_prediction(output)
        output = super().postprocess(output)
        return output


class DirectCausalLMClassifier(HFClassifier, InstructCausalLM):
    def __init__(self, model_name, labels, **kwargs) -> None:
        super().__init__(model_name, labels, **kwargs)
    
    def default_input_prompt(self):
        return "Text: {input}"
    
    def default_user_prompt(self):
        return "{label_type}:"

    def get_scores_for_labels(self, model_input, labels, model, tokenizer):
        model_input = model_input.strip() + " "
        # random.shuffle(labels)
        _labels = labels
        inputs = [f"{model_input}{l}" for l in _labels]

        # Get encodings
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        input_enc = tokenizer([model_input], return_tensors="pt")
        inputs_enc = tokenizer(inputs, return_tensors="pt", padding="longest")

        # print(223, [tokenizer.convert_ids_to_tokens(ids) for ids in inputs_enc.input_ids])
        target_enc = inputs_enc.input_ids.clone()
        target_enc = (
            target_enc
            - (100 + tokenizer.pad_token_id)
            * (target_enc == tokenizer.pad_token_id).long()
        )

        with torch.no_grad():
            inputs_enc = inputs_enc.to(model.device)
            model_output = model(
                attention_mask=inputs_enc.attention_mask,
                input_ids=inputs_enc.input_ids,
                return_dict=True,
            )

        logits = model_output.logits[:, :-1]
        target_enc = target_enc[:, 1:]
        logits = logits.flatten(0, 1)
        target_enc = target_enc.flatten(0, 1)

        # Compute the log probabilities associated with each of the labels
        logits = logits.type(torch.float32).to(target_enc.device)
        labels_log_probs = F.cross_entropy(logits, target_enc, reduction="none")

        # Sum log probs for each of the (input, label) pair
        labels_scores = labels_log_probs.view(len(labels), -1)
        input_len = input_enc.input_ids.shape[1]

        labels_scores = labels_scores[:, input_len - 2 :]
        labels_scores = labels_scores.sum(dim=-1)
        # Note: Label log probabilities are positive (due to the internals of pytorch's
        # cross entropy). To obtain the "logits", we need to multiply by -1.
        labels_scores = labels_scores * -1
        labels_scores = labels_scores.exp().detach().numpy()
        probs = labels_scores / labels_scores.sum()
        dist = [(label,prob) for label, prob in zip(_labels, probs)]
        label = labels[np.argmax(labels_scores)]
        output = dict(text=self.input_data, distribution=dist, label=label)
        return output

    @memoize(ignore_kwargs=["model_path"])
    def generate_cached(
        self, model_name, model_input, labels, memoizer_ignore_cache=False, **kwargs
    ):
        model_kwargs = self.get_model_kwargs()
        model = self.load_model(**model_kwargs)
        tokenizer = self.load_tokenizer()
        return self.get_scores_for_labels(model_input, labels, model, tokenizer)

    def postprocess(self, output):
        if isinstance(output, dict) and "label" in output:
            output = output["label"]
        output = super().postprocess(output)
        return output


class InContextClassifier(BaseClassifier):
    def __init__(
        self,
        model_name,
        labels,
        confidence_threshold=None,
        memory_per_class=0,
        **kwargs,
    ) -> None:
        super().__init__(model_name, labels, **kwargs)
        if confidence_threshold is None:
            confidence_threshold = min(2 * (1. / len(labels)), .7)
        self.confidence_threshold = confidence_threshold
        self.memory_per_class = memory_per_class
        self.memory = []

    def get_prompt_args(self):
        args = super().get_prompt_args()
        args["context"] = self.context_prompt()
        return args

    def default_input_prompt(self):
        return "{context}Text: {input}"

    def default_user_prompt(self):
        return "{label_type}:"

    def recall_samples(self):
        confident_samples = []
        sample_counts = {}
        
        for sample in self.memory:
            category = sample["label"]
            add_to_memory = True

            if "distribution" in sample:
                probs = sample["distribution"]
                probs = sorted(probs, key=lambda x: x[1], reverse=True)
                if probs[0][1] < self.confidence_threshold:
                    add_to_memory = False

            if add_to_memory:
                class_count = sample_counts.get(category, 0)
                # make sure samples remain reasonably uniform across labels
                max_samples = min([sample_counts.get(l, 0) for l in self.labels])
                max_samples = min(max_samples+1, self.memory_per_class)

                if class_count < max_samples:
                    confident_samples.append(sample)
                    class_count += 1
                    sample_counts[category] = class_count

        return confident_samples

    def context_prompt(self):
        samples = self.recall_samples()

        if samples:
            random.shuffle(samples)
            logger.debug(f"Using {len(samples)} samples from memory")

            prompt = [
                f"Text: {s['text']}\nCategory: {s['label']}" for s in samples
            ]
            prompt = "\n\n".join(prompt)
            prompt += "\n\n"
        else:
            prompt = ""

        return prompt

    def postprocess(self, output):
        if isinstance(output, dict):
            self.memory.append(output)
            output = output["label"]
        output = super().postprocess(output)
        return output


class InContextDirectClassifier(InContextClassifier, DirectCausalLMClassifier):
    def __init__(self, model_name, labels, **kwargs) -> None:
        super().__init__(model_name, labels, **kwargs)


class InstructCausalLMClassifier(InstructTunedClassifier, InstructCausalLM):
    def __init__(self, model_name, labels, **kwargs) -> None:
        super().__init__(model_name, labels, **kwargs)


class AlpacaClassifier(Alpaca, InstructCausalLMClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class VicunaClassifier(Vicuna, InstructCausalLMClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class LlamaChatClassifier(InstructTunedClassifier, LlamaChat):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def default_system_prompt(self):
        return None


class OpenAIClassifier(OpenAIChat, InstructTunedClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
