# PCB Benchmark

基于OpenRouter的PCB知识评估工具，支持单选题和问答题两种评估模式。

## 功能

- **单选题评估**：客观题自动评分，准确率统计
- **问答题评估**：基于F1分数的主观题评估
- **多模型比较**：并行评估多个模型性能
- **结果保存**：自动生成详细的评估报告

## 环境要求

- Python 3.8+
- OpenRouter API密钥

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

#### 2.1 获取API密钥

1. 访问 [OpenRouter官网](https://openrouter.ai)
2. 注册账户并获取API密钥

#### 2.2 配置环境变量

编辑 `config/.env` 文件，添加您的API密钥：
```bash
OPENROUTER_API_KEY=sk-your-api-key-here
```

### 3. 运行评估

#### 单选题评估

```bash
# 单个模型
python eval_scq.py --model gpt-4o

# 多个模型
python eval_scq.py --model "gpt-4o,claude-opus-4-20250514"

# 限制题目数量
python eval_scq.py --model gpt-4o --max-questions 10
```

#### 问答题评估

```bash
# 单个模型
python eval_qa.py --model gemini-2.5-pro

# 多个模型
python eval_qa.py --model "deepseek-v3-250324,gpt-4o"
```

## 支持的模型

支持OpenRouter平台上的所有模型，包括但不限于：
- GPT系列：`gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- Claude系列：`claude-opus-4-20250514`, `claude-3-5-sonnet-20241022`
- Gemini系列：`gemini-2.5-pro`, `gemini-1.5-pro`
- 国产模型：`deepseek-v3-250324`, `qwen-plus`, `yi-large`

完整模型列表请查看 [OpenRouter官网](https://openrouter.ai)。

## 配置文件

### config.yaml 详细配置

```yaml
# LLM配置
llm:
  base_url: "https://openrouter.ai/api/v1"    # OpenRouter API地址
  temperature: 0.0                       # 温度参数，0.0-1.0
  max_tokens: 2048                       # 最大生成token数
  timeout: 60                            # 请求超时时间（秒）
  max_retries: 3                         # 最大重试次数
  retry_delay: 1.0                       # 重试间隔（秒）
  default_model: "gpt-4o"                # 默认模型

# 数据配置
data:
  raw_path: "data/raw"                   # 原始数据路径
  processed_path: "data/processed"       # 处理后数据路径
  single_choice_questions_file: "single_choice_questions.jsonl"  # 单选题文件
  qa_questions_file: "qa_questions.jsonl"                      # 问答题文件

# 输出配置
output:
  results_dir: "results"                 # 结果保存目录

# 评测配置
evaluation:
  save_progress: true                    # 是否保存进度
  max_questions: null                    # 最大问题数，null表示全部
  
  # 并行模式配置
  parallel:
    show_progress: true                  # 是否显示实时进度
    sort_results: true                   # 是否按问题索引排序结果

# 系统提示词配置
system_prompt: |
  你是一个专业的PCB设计和电子工程专家，具有丰富的电路设计、PCB布局、信号完整性分析等方面的知识。
  
  请根据你的专业知识回答以下PCB相关的单选题。对于每个问题：
  1. 仔细分析问题和各个选项
  2. 运用你的PCB设计知识进行推理
  3. 选择最正确的答案
  4. 只需要回答选项字母（A、B、C、D或E），不需要解释

# 问答题系统提示词
qa_system_prompt: |
  你是一个专业的PCB设计和电子工程专家，具有丰富的电路设计、PCB布局、信号完整性分析等方面的知识。
  
  请根据你的专业知识回答以下PCB相关的问答题。对于每个问题：
  1. 仔细分析问题的要求
  2. 运用你的PCB设计知识进行分析
  3. 提供准确、详细、专业的答案
  4. 答案应该简洁明了，重点突出
```

### 配置自定义

#### 提示词自定义

您可以根据需要修改 `system_prompt` 和 `qa_system_prompt` 来定制模型的行为。

#### 并行评估配置

```yaml
evaluation:
  parallel:
    show_progress: true    # 显示实时进度条
    sort_results: true     # 按问题索引排序结果
```

#### 超时和重试配置

```yaml
llm:
  timeout: 60           # 单次请求超时时间
  max_retries: 3        # 失败重试次数
  retry_delay: 1.0      # 重试间隔
```

## 结果文件

评估结果按以下结构组织：
```
results/
├── scq/                              # 单选题结果
│   ├── gpt-4o/
│   │   ├── results_20250706_113001.jsonl
│   │   └── report_20250706_113001.json
│   └── claude-opus-4-20250514/
│       ├── results_20250706_114502.jsonl
│       └── report_20250706_114502.json
└── qa/                               # 问答题结果
    ├── gemini-2.5-pro/
    │   ├── results_20250706_115806.jsonl
    │   └── report_20250706_115806.json
    └── deepseek-v3-250324/
        ├── results_20250706_115839.jsonl
        └── report_20250706_115839.json
```

## 项目结构

```
pcb_benchmark/
├── config/
│   ├── config.yaml        # 主配置文件
│   └── .env               # 环境变量
├── data/
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后数据
│   └── preprocess.py      # 数据预处理脚本
├── results/               # 评估结果
├── evaluator.py           # 评估器核心逻辑
├── utils.py               # 工具函数
├── runner.py              # 评估流程控制
├── eval_scq.py            # 单选题评估入口
├── eval_qa.py             # 问答题评估入口
└── requirements.txt       # 依赖包
```

## 常见问题

### 模型名称错误

如果使用了不存在的模型名称，OpenRouter API会返回具体的错误信息：
```
Error code: 400 - {'error': {'message': 'Incorrect model ID. Please request to view the model page or you do not have permission to use this model...'}}
```

### 环境变量未设置

如果未设置 `OPENROUTER_API_KEY`，程序会提示：
```
❌ 未找到OpenRouter API密钥，请设置环境变量 OPENROUTER_API_KEY
```

### 数据文件不存在

如果数据文件不存在，程序会直接报错并停止运行。请确保 `data/processed/` 目录下有相应的数据文件。


### 评估的模型：
openai/gpt-5   NA
openai/gpt-4o  NA
anthropic/claude-opus-4.1   NA
google/gemini-2.5-pro   NA
deepseek/deepseek-chat-v3.1   671B 
deepseek/deepseek-r1-0528   671B
qwen/qwen-2.5-7b-instruct   7B
meta-llama/llama-4-maverick   400B