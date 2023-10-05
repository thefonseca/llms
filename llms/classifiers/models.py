import logging
from pprint import pformat
import random

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
    FalconChat,
    MistralInstruct,
)
from ..models.openai import OpenAIChat
from ..utils.memoizer import memoize

logger = logging.getLogger(__name__)


class BaseClassifier(BaseLM):
    def __init__(
        self, model_name, labels, input_type="text", label_type="category", **kwargs
    ) -> None:
        super().__init__(model_name, **kwargs)
        self.labels = labels
        self.label_type = label_type
        self.input_type = input_type

    def get_prompt_args(self):
        label_type = self.label_type.capitalize()
        input_type = self.input_type.capitalize()
        labels = self.labels
        if isinstance(labels, (list, tuple, dict)):
            labels = [f"- {label}" for label in labels]
            labels = "\n".join(labels)
        return dict(labels=labels, label_type=label_type, input_type=input_type)

    def postprocess(self, output):
        output = super().postprocess(output)
        # Labels can be specified as a map from a natural language label to a symbol.
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
    def __init__(self, model_name, labels, multi_label=False, **kwargs) -> None:
        super().__init__(model_name, labels, **kwargs)
        self.multi_label = multi_label

    def default_context_prompt(self):
        return (
            "Classify the text below into one of the following categories:\n{labels}"
            "\nBe concise and write only the category name."
        )

    def default_input_prompt(self):
        return "{input_type}: {input}"

    def default_user_prompt(self):
        return "{label_type}:"

    def fix_prediction(self, output):
        if isinstance(self.labels, (list, tuple, dict)):
            target_labels = [x.lower() for x in self.labels]
        else:
            target_labels = self.labels.lower()

        if output == "":
            logger.warning(f"Prediction is empty")

        elif output.lower() not in target_labels:
            # if not self.multi_label:
            #     output = output.split(",")[0]
            output = output.strip()
            found_labels = list(
                set([label for label in self.labels if label.lower() in output.lower()])
            )
            if len(found_labels) == 1:
                logger.warning(f'Fixing prediction: "{output}" => "{found_labels[0]}"')
                output = found_labels[0]

            if output.lower() not in target_labels:
                for label in self.labels:
                    # some models output truncated labels
                    if output.lower() == label.lower()[: len(output)]:
                        logger.warning(f'Fixing prediction: "{output}" => "{label}"')
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


class CausalLMClassifier(HFClassifier, InstructCausalLM):
    def __init__(
        self,
        model_name,
        labels,
        prior_normalization=False,
        noisy_channel=False,
        **kwargs,
    ) -> None:
        self.noisy_channel = noisy_channel
        super().__init__(model_name, labels, **kwargs)
        self.prior_normalization = prior_normalization
        self.prior_label_probs = {}

    def default_input_prompt(self):
        return "{input_type}: {input}"

    def default_user_prompt(self):
        return "{label_type}:"

    def process_generation_kwargs(self, **generation_kwargs):
        kwargs = super().process_generation_kwargs(**generation_kwargs)
        if "labels" not in kwargs:
            kwargs["labels"] = self.labels
        if "label_type" not in kwargs:
            kwargs["label_type"] = self.label_type
        if "input_type" not in kwargs:
            kwargs["input_type"] = self.input_type
        if "prior_normalization" not in kwargs:
            kwargs["prior_normalization"] = self.prior_normalization
        if "noisy_channel" not in kwargs:
            kwargs["noisy_channel"] = self.noisy_channel
        return kwargs

    def _get_label_probs(self, encoded_input, label_log_probs):
        input_len = encoded_input.input_ids.shape[1]
        start_idx = max(0, input_len - 2)
        labels_scores = label_log_probs[:, start_idx:]
        labels_scores = labels_scores.sum(dim=-1)
        # mask = labels_scores != 0
        # labels_scores = (labels_scores * mask).sum(dim=-1) / mask.sum(dim=-1)

        # Note: Label log probabilities are positive (due to the internals of pytorch's
        # cross entropy). To obtain the "logits", we need to multiply by -1.
        labels_scores = labels_scores * -1
        labels_scores = labels_scores.exp()
        label_probs = labels_scores / labels_scores.sum()
        return label_probs.detach().numpy()

    def get_scores_for_labels(
        self, model_input, labels, model, tokenizer, prior_normalization, noisy_channel
    ):
        if isinstance(labels, dict):
            labels = sorted(list(labels.keys()))

        prior_label_probs = None

        if noisy_channel:
            model_inputs_ = [f"{self._context_prompt}\n\n{l}:" for l in labels]
            labels_ = [self.input_data]
            model_inputs = model_inputs_
        else:
            model_inputs = [model_input]
            labels_ = labels

        if not noisy_channel and prior_normalization:
            prior_label_probs = [
                self.prior_label_probs[l] for l in labels if l in self.prior_label_probs
            ]
            if len(prior_label_probs) < len(labels):
                prior_label_probs = None
                label_type = self.label_type.capitalize()
                model_inputs.append(f"{label_type}:")

        model_inputs = [x.strip() + " " for x in model_inputs]
        inputs_and_labels = [f"{x}{l}" for x in model_inputs for l in labels_]

        # Get encodings
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        inputs_enc = [tokenizer(x, return_tensors="pt") for x in model_inputs]
        inputs_and_labels_enc = tokenizer(
            inputs_and_labels, return_tensors="pt", padding="longest"
        )
        target_enc = inputs_and_labels_enc.input_ids.clone()
        target_enc = (
            target_enc
            - (100 + tokenizer.pad_token_id)
            * (target_enc == tokenizer.pad_token_id).long()
        )

        with torch.no_grad():
            inputs_and_labels_enc = inputs_and_labels_enc.to(model.device)
            model_output = model(
                attention_mask=inputs_and_labels_enc.attention_mask,
                input_ids=inputs_and_labels_enc.input_ids,
                return_dict=True,
            )

        logits = model_output.logits[:, :-1]
        target_enc = target_enc[:, 1:]
        logits = logits.flatten(0, 1)
        target_enc = target_enc.flatten(0, 1)

        # Compute the log probabilities associated with each of the labels
        logits = logits.type(torch.float32).to(target_enc.device)
        log_probs = F.cross_entropy(logits, target_enc, reduction="none")
        log_probs = log_probs.view(len(model_inputs), len(labels_), -1)

        if noisy_channel:
            log_probs = torch.transpose(log_probs, 0, 1)

        label_probs = self._get_label_probs(inputs_enc[0], log_probs[0])

        if not noisy_channel and prior_normalization:
            if prior_label_probs is None:
                prior_label_probs = self._get_label_probs(inputs_enc[1], log_probs[1])
                for idx, label in enumerate(labels):
                    self.prior_label_probs[label] = prior_label_probs[idx]
                logger.info(
                    f"Prior label probabilities:\n{pformat(self.prior_label_probs)}"
                )
            label_probs /= prior_label_probs

        dist = [(label, prob) for label, prob in zip(labels, label_probs)]
        label = labels[np.argmax(label_probs)]
        output = dict(text=self.input_data, distribution=dist, label=label)
        return output

    @memoize(ignore_kwargs=["model_path", "max_memory"])
    def generate_cached(
        self,
        model_name,
        model_input,
        labels,
        prior_normalization=False,
        noisy_channel=False,
        memoizer_ignore_cache=False,
        **kwargs,
    ):
        model_kwargs = self.get_model_kwargs()
        model = self.load_model(**model_kwargs)
        tokenizer = self.load_tokenizer()
        return self.get_scores_for_labels(
            model_input, labels, model, tokenizer, prior_normalization, noisy_channel
        )

    def postprocess(self, output):
        if isinstance(output, dict) and "label" in output:
            output = output["label"]
        output = super().postprocess(output)
        return output


class DynamicContextClassifier(BaseClassifier):
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
            confidence_threshold = min(2 * (1.0 / len(labels)), 0.7)
        self.confidence_threshold = confidence_threshold
        self.memory_per_class = memory_per_class
        self.memory = []

    def get_prompt_args(self):
        args = super().get_prompt_args()
        args["context"] = self.context_prompt()
        return args

    def default_input_prompt(self):
        return "{context}{input_type}: {input}"

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
                max_samples = min(max_samples + 1, self.memory_per_class)

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
                f"{{input_type}}: {s['text']}\{{label_type}}: {s['label']}"
                for s in samples
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


class DynamicContextDirectClassifier(DynamicContextClassifier, CausalLMClassifier):
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

    def fix_prediction(self, output):
        output = output.replace("] ", "").strip()
        return super().fix_prediction(output)


class FalconChatClassifier(FalconChat, InstructCausalLMClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class MistralInsructClassifier(MistralInstruct, InstructCausalLMClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class OpenAIClassifier(InstructTunedClassifier, OpenAIChat):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def postprocess(self, output):
        output = output["choices"][0]["message"]["content"]
        output = super().postprocess(output)
        return output
