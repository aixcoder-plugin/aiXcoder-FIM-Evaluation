# aiXcoder FIM Code Generation LLM Evaluation

这是用于评估FIM代码生成的工具, 用于在java/python/cpp/jsvascript四种语言的数据集上进行生成和评估任务。

## 数据集与评估指标简介
对于java/python/cpp/jsvascript四种语言，提供代码块的上文和下文信息，需要预测中间填充部分。
评估指标包括Exact Match、BLEU-4、CODE-BLEU、Length(Pred/Ref)
- Exact Match
- BLEU-4
- CODE-BLEU
- Length(Pred/Ref)

## 环境要求
主要的环境依赖为：

- Python 3.8 or higher
- PyTorch 2.1.0 or higher
- sentencepiece 0.2.0 or higher
- transformers 4.34.1 or higher (if run inference by transformers library)

在支持CUDA环境的宿主机或者容器内，执行以下命令安装环境依赖项：

```bash
conda create -n aixcoder-evaluation python=3.11
conda activate aixcoder-evaluation
pip install -r requirements.txt
```

`requirements.txt` 列举了所有的依赖项及其版本号。

如果想要加快推理速度，我们强烈建议安装 FlashAttention 库（可选）。在确定您的芯片版本与CUDA版本支持FlashAttention 的条件下，可通过以下步骤进行安装：

```bash
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=8 python setup.py install
```

## 用法
### 数据集准备
```bash
cd datasets
tar zxvf *.tar.gz
```

### 生成
下面是生成任务的示例。
python run_inference.py --model aiXcoder/aixcoder-7b-base --language java

- `--model`huggingface上的模型名称,
### 目前可以对于huggingface上的四种模型进行FIM生成
    - deepseek-ai/deepseek-coder-6.7b-base
    - aiXcoder/aixcoder-7b-base
    - codellama/CodeLlama-7b-hf
    - bigcode/starcoder2-7b
    也可以设置本地已经下载的模型权重文件
- `--language`数据集语言
    - 支持python java cplus javascript四种语言, 可以单独设置一种语言, 也可以同时设置多种语言, 多个语言之间通过空格分割

- `--output_dir`生成结果输出路径, 默认保存在当前目录下的output_dir文件夹下

- `--device`设置使用的显卡, 默认cuda

- `--torch_dtype`设置精度, 默认bf16, 可设置为"fp32", "fp16", "bf16"

- `--attn_implementation`设置使用FlashAttention, 默认设置True, 若不支持FlashAttention将此项设置为False

- `--gen_len`设置最大生成长度, 默认512

- `--max_len`设置最大序列长度, 默认16384

### 评估
下面是评估任务的示例。
python run_evaluate.py
- `--language`要评测的语言
    - 支持python java cplus javascript四种语言, 可以单独设置一种语言, 也可以同时设置多种语言, 多个语言之间通过空格分割

- `--result_path`评估结果输出路径, 默认保存在当前目录下的output_dir文件夹下
会产生两个文件后缀为_scored.jsonl和_statistics.txt
其中_statistics.txt中记录了各种不同Task Type的各项评估结果以及总结果平均值




