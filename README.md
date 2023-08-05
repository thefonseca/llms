# LLMs

This is a tool to evaluate of large language models on NLP tasks. It implements a common API for traditional encoder-decoder and prompt-based large language models, as well as APIs such as OpenAI and Cohere.

Currently, these functionalities are available:
- Prompting and truncation logic
- Support for vanilla LLMs ([OPT](https://arxiv.org/abs/2205.01068), [LLaMa](https://github.com/facebookresearch/llama)) and instruction-tuned models ([T0](https://github.com/bigscience-workshop/t-zero), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)) 
- Evaluation based on [🤗 Datasets](https://github.com/huggingface/datasets) or CSV files
- Memoization: inference outputs are cached on disk
- Parallelized computation of metrics

## Setup
```bash
git clone https://github.com/thefonseca/llms.git
cd llms && pip install -e .
```

## Classification examples
```
python -m llms.classifiers.evaluation \
--model_name llama-2-7b-chat 
--model_checkpoint_path path_to_llama2_checkpoint 
--model_dtype fp16 
--dataset_name imdb 
--split test 
--source_key text 
--target_key label 
--model_labels "{'Positive':1,'Negative':0}" 
--max_samples 1000
```

## Summarization examples
Evaluating [BigBird](https://github.com/google-research/bigbird) on [PubMed](https://huggingface.co/datasets/scientific_papers) validation split, and saving the results on the `output` folder:

```bash
python -m llms.summarizers.evaluation \
--dataset_name scientific_papers \
--dataset_config pubmed \
--split validation \
--source_key article \
--target_key abstract \
--max_samples 1000 \
--model_name google/bigbird-pegasus-large-pubmed \
--output_dir output
```
where `--model_name` is a [huggingface model identifier](https://huggingface.co/models).

Evaluating [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) (float16) on [arXiv](https://huggingface.co/datasets/scientific_papers) validation split:

```bash
python -m llms.summarizers.evaluation \
--arxiv_id https://arxiv.org/abs/2304.15004v1 \
--model_name alpaca-7b \
--model_checkpoint_path path_to_alpaca_checkpoint \
--budget 7 \
--budget_unit sentences \
--model_dtype fp16 \
--output_dir output
```

Notes:
- `--budget` controls length of instruct-tuned summaries (by default, in sentences).
- `--model_checkpoint_path` allows changing checkpoint folder while keeping the cache
key (`--model_name`) constant.

Evaluating [ChatGPT API](https://platform.openai.com/docs/api-reference/chat) on [arXiv](https://huggingface.co/datasets/scientific_papers) validation split:

```bash
export OPENAI_API_KEY=<your_api_key>
python -m llms.summarizers.evaluation \
--dataset_name scientific_papers \
--dataset_config arxiv \
--split validation \
--source_key article \
--target_key abstract \
--max_samples 1000 \
--model_name gpt-3.5-turbo \
--output_dir output
```

Evaluating summary predictions from a CSV file:

```bash
python -m llms.summarizers.evaluation \
--dataset_name scientific_papers \
--dataset_config arxiv \
--split validation \
--source_key article \
--target_key abstract \
--prediction_path path_to_csv_file \
--prediction_key prediction \
--max_samples 1000 \
--output_dir output
```