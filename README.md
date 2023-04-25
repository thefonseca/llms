# Summarizers

Summarizers is a tool to evaluate of summarization models. It implements a common API for traditional encoder-decoder and prompt-based large language models, as well as APIs such as OpenAI and Cohere.

Currently, these functionalities are available:
- Summarization prompting and truncation logic
- Support for vanilla LLMs ([OPT](https://arxiv.org/abs/2205.01068), [LLaMa](https://github.com/facebookresearch/llama)) and instruction-tuned models ([T0](https://github.com/bigscience-workshop/t-zero), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)) 
- Evaluation based on [🤗 Datasets](https://github.com/huggingface/datasets) or CSV files
- Memoization: inference outputs are cached on disk
- Parallelized computation of metrics

## Setup
```bash
git clone https://github.com/thefonseca/summarizers.git
cd summarizers && pip install -r requirements.txt
```

## Examples
Evaluating [BigBird](https://github.com/google-research/bigbird) on [PubMed](https://huggingface.co/datasets/scientific_papers) validation split, and saving the results on the `output` folder:

```bash
python evaluation.py \
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
python evaluation.py \
--dataset_name scientific_papers \
--dataset_config arxiv \
--split validation \
--source_key article \
--target_key abstract \
--max_samples 1000 \
--model_name path_to_alpaca_checkpoint \
--budget 7 \
--budget_unit sentences \
--model_dtype fp16 \
--output_dir output
```

Notes:
- `budgetbudget` controls length of instruct-tuned summaries (by default, in sentences).
- `path_to_alpaca_checkpoint` has to contain the string "alpaca" so that the correct summarizer class `AlpacaSummarizer` is used.

Evaluating [ChatGPT API](https://platform.openai.com/docs/api-reference/chat) on [arXiv](https://huggingface.co/datasets/scientific_papers) validation split:

```bash
export OPENAI_API_KEY=<your_api_key>
python evaluation.py \
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
python evaluation.py \
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
