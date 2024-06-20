# aiXcoder FIM Code Generation LLM Evaluation

This is a tool for evaluating FIM code generation for generating and evaluating tasks on datasets in four languages: java/python/cpp/jsvascript.

## Introduction to datasets and evaluation metrics
For Java, Python, CPP, and JSVASCRIPT languages, to provide the upper and lower information of the code block, you need to predict the middle filling.
Evaluation metrics include:Exact Match、BLEU-4、CODE-BLEU、Length(Pred/Ref)
- Exact Match
- BLEU-4
- CODE-BLEU
- Length(Pred/Ref)

## Environment Requirements
To run the model inference and evaluatiaon code, you'll need the following environment setup:

- Python 3.8 or higher
- PyTorch 2.1.0 or higher
- sentencepiece 0.2.0 or higher
- transformers 4.34.1 or higher (if run inference by transformers library)

Please ensure all dependencies are installed using the following command:

```bash
conda create -n aixcoder-evaluation python=3.11
conda activate aixcoder-evaluation
pip install -r requirements.txt
```

`requirements.txt` listed all necessary libraries and their versions.

To achieve faster inference speeds, especially for large models, we recommend installing `flash attention`. `Flash attention` is an optimized attention mechanism that significantly reduces computation time for transformer-based models without sacrificing accuracy.

Before proceeding, ensure your environment meets the CUDA requirements as `flash attention` leverages GPU acceleration. Follow these steps to install `flash attention`:

```bash
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=8 python setup.py install
```

## Usage
### Datasets preparation
```bash
cd datasets
tar zxvf *.tar.gz
```
### Generation
Here's an example of a generate task.
python run_inference.py --model aiXcoder/aixcoder-7b-base --language java

- `--model`model name on huggingface,
### Currently, FIM generation can be performed for four models on the huggingface
    - deepseek-ai/deepseek-coder-6.7b-base
    - aiXcoder/aixcoder-7b-base
    - codellama/CodeLlama-7b-hf
    - bigcode/starcoder2-7b
    You can also set the model weight file that has been downloaded locally
- `--language`Dataset language
    - Support Python Java Cplus JavaScript four languages, you can set a language separately, you can also set multiple languages at the same time, and multiple languages are separated by spaces

- `--output_dir`The output path of the generated result is saved in the output_dir folder in the current directory by default

- `--device`Set the cuda used, default cuda

- `--torch_dtype`Set the precision, default bf16, can be set to:"fp32", "fp16", "bf16"

- `--attn_implementation`The setting uses FlashAttention, default True, if you don't support FlashAttention, set this to False

- `--gen_len`Set max generate length, default 512

- `--max_len`Set `max_new_tokens`, default 16384

### Evaluation
Here's an example of a evaluate task.
python run_evaluate.py
- `--language`The language to be evaluated
    - Support Python Java Cplus JavaScript four languages, you can set a language separately, you can also set multiple languages at the same time, and multiple languages are separated by spaces

- `--result_path`By default, the output path of the evaluation results is stored in the output_dir folder in the current directory
Two files are generated with the suffix _scored.jsonl and _statistics.txt
The results of each assessment for each Task Type and the average of the total results are recorded in the _statistics.txt




