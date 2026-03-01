#!/usr/bin/env python3
"""
工具函数模块
包含LLM客户端、数据处理、文件操作等通用功能
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import re
from openai import OpenAI, AsyncOpenAI

# 数据处理相关
import jsonlines

# 配置管理
import yaml
from dotenv import load_dotenv

@dataclass
class LLMConfig:
    """LLM配置类"""
    model_name: str
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 60

class LLMClient:
    """统一LLM客户端 - 基于AiHubMix"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout
        )
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """生成响应"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # 检查响应类型
            if isinstance(response, str):
                print(f"⚠️ API返回字符串响应: {response[:100]}...")
                return response
            
            # 检查响应对象结构
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            else:
                print(f"⚠️ 响应对象缺少choices属性: {type(response)}")
                print(f"响应内容: {response}")
                return None
            
        except Exception as e:
            print(f"❌ LLM调用失败: {e}")
            print(f"错误类型: {type(e)}")
            return None

class AsyncLLMClient:
    """异步LLM客户端 - 基于AiHubMix"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
    
    async def __aenter__(self):
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.close()
    
    async def generate_response_async(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """异步生成响应"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # 检查响应类型
            if isinstance(response, str):
                print(f"⚠️ API返回字符串响应: {response[:100]}...")
                return response
            
            # 检查响应对象结构
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            else:
                print(f"⚠️ 响应对象缺少choices属性: {type(response)}")
                print(f"响应内容: {response}")
                return None
            
        except Exception as e:
            print(f"❌ 异步LLM调用失败: {e}")
            print(f"错误类型: {type(e)}")
            return None

def create_llm_config(llm_config: Dict[str, Any], model_name: str, keys: Dict[str, str]) -> LLMConfig:
    """创建LLM配置"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("❌ 未找到AiHubMix API密钥，请设置环境变量 OPENROUTER_API_KEY")
    
    return LLMConfig(
        model_name=model_name,
        api_key=api_key,
        base_url=llm_config.get("base_url", "https://openrouter.ai/api/v1"),
        temperature=llm_config.get("temperature", 0.0),
        max_tokens=llm_config.get("max_tokens", 2048),
        timeout=llm_config.get("timeout", 60)
    )

def create_llm_configs_for_models(config: Dict[str, Any], model_names: List[str]) -> List[LLMConfig]:
    """为多个模型创建LLM配置"""
    configs = []
    llm_config = config.get('llm', {})
    
    for model_name in model_names:
        try:
            llm_cfg = create_llm_config(llm_config, model_name, {})
            configs.append(llm_cfg)
        except Exception as e:
            print(f"❌ 创建模型 {model_name} 配置失败: {e}")
    
    return configs

# 文件操作函数
def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    with jsonlines.open(file_path, mode='r') as reader:
        return list(reader)

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """保存JSONL文件"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(data)
    print(f"✅ 成功保存 {len(data)} 条数据到 {file_path}")

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_env_file(env_path: str) -> bool:
    """加载环境变量文件"""
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"✅ 已加载环境变量文件: {env_path}")
        return True
    else:
        print(f"⚠️ 环境变量文件不存在: {env_path}")
        return False

# 数据处理函数
def load_evaluation_questions(config: Dict[str, Any], max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载评估问题"""
    data_path = config['data']['processed_path']
    questions_file = config['data']['single_choice_questions_file']
    file_path = os.path.join(data_path, questions_file)
    
    questions = load_jsonl(file_path)
    print(f"✅ 成功加载 {len(questions)} 条数据从 {file_path}")
    
    if max_questions:
        questions = questions[:max_questions]
        print(f"限制评估问题数量为: {max_questions}")
    
    return questions

def load_qa_questions(config: Dict[str, Any], max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载问答题"""
    data_path = config['data']['processed_path']
    questions_file = config['data']['qa_questions_file']
    file_path = os.path.join(data_path, questions_file)
    
    questions = load_jsonl(file_path)
    print(f"✅ 成功加载 {len(questions)} 条问答题数据从 {file_path}")
    
    if max_questions:
        questions = questions[:max_questions]
        print(f"限制评估问题数量为: {max_questions}")
    
    return questions

# 文本处理函数
def format_single_choice_prompt(question_data: Dict[str, Any]) -> str:
    """格式化单选题提示"""
    question = question_data.get('question', '')
    options = question_data.get('options', [])
    
    prompt = f"问题：{question}\n\n选项：\n"
    for i, option in enumerate(options):
        letter = chr(ord('A') + i)
        prompt += f"{letter}. {option}\n"
    
    prompt += "\n请选择正确答案的字母："
    return prompt

def extract_choice_from_response(response: str) -> Optional[str]:
    """从响应中提取选择答案"""
    if not response:
        return None
    
    # 匹配单独的选项字母
    pattern = r'\b([A-E])\b'
    matches = re.findall(pattern, response.upper())
    
    if matches:
        return matches[0]
    
    # 匹配"答案是A"或"选择A"等模式
    pattern = r'(?:答案是|选择|答案为|选项)\s*([A-E])'
    matches = re.findall(pattern, response.upper())
    
    if matches:
        return matches[0]
    
    return None

def get_evaluation_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """获取评估摘要"""
    total_questions = len(results)
    correct_answers = sum(1 for r in results if r.get('correct', False))
    accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
    
    # 统计预测情况
    no_prediction = sum(1 for r in results if not r.get('prediction'))
    valid_predictions = sum(1 for r in results if r.get('prediction') and r.get('prediction') in ['A', 'B', 'C', 'D', 'E'])
    invalid_prediction = total_questions - no_prediction - valid_predictions
    
    return {
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'accuracy': accuracy,
        'no_prediction': no_prediction,
        'valid_predictions': valid_predictions,
        'invalid_prediction': invalid_prediction
    }

def generate_output_filenames(config: Dict[str, Any], suffix: str = "", task_type: str = "scq") -> tuple:
    """生成输出文件名 - 按任务和模型分文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = config['output']['results_dir']
    
    # 从suffix中提取模型名（suffix格式: _modelname 或 _qa_modelname）
    if suffix.startswith("_qa_"):
        model_name = suffix[4:]  # 去掉 "_qa_" 前缀
    elif suffix.startswith("_"):
        model_name = suffix[1:]  # 去掉 "_" 前缀
    else:
        model_name = "unknown"
    
    # 构建文件夹路径：results/task_type/model_name/
    task_dir = os.path.join(results_dir, task_type, model_name)
    Path(task_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成文件名（包含时间戳）
    results_file = os.path.join(task_dir, f"results_{timestamp}.jsonl")
    report_file = os.path.join(task_dir, f"report_{timestamp}.json")
    
    return results_file, report_file

def execute_evaluation(evaluator, questions, config, results_file):
    """并行评估执行函数"""
    print("🚀 启动并行评测模式...")
    
    results = evaluator.evaluate_parallel_mode(
        questions,
        save_progress=config['evaluation'].get('save_progress', True),
        progress_file=results_file
    )
    return results

def print_evaluation_completion(results_file, report_file, report):
    """打印评估完成信息"""
    print("\n✅ 评估完成!")
    print(f"📊 结果文件: {results_file}")
    print(f"📋 报告文件: {report_file}")
    
    # 根据报告类型显示不同的摘要信息
    summary = report.get('evaluation_summary', {})
    if 'accuracy' in summary:
        # 单选题评估
        print(f"🎯 总体准确率: {summary.get('accuracy', 0):.2%}")
    elif 'average_f1' in summary:
        # 问答题评估
        print(f"🎯 平均F1分数: {summary.get('average_f1', 0):.3f}")
        print(f"🎯 平均精确率: {summary.get('average_precision', 0):.3f}")
        print(f"🎯 平均召回率: {summary.get('average_recall', 0):.3f}")

def parse_model_names(model_str: str) -> List[str]:
    """解析模型名称字符串"""
    return [name.strip() for name in model_str.split(',') if name.strip()] 