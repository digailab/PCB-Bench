# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("⏳ Waiting for debugger attach...")
# debugpy.wait_for_client() 
"""
评估运行器模块
负责协调评估流程的执行
"""

import argparse
import os
import sys
from typing import List, Optional, Dict, Any

from utils import (
    load_jsonl,
    load_yaml_config,
    load_env_file,
    create_llm_config,
    create_llm_configs_for_models,
    generate_output_filenames,
    execute_evaluation,
    print_evaluation_completion,
    parse_model_names
)
from evaluator import PCBBenchmarkEvaluator, PCBQAEvaluator, FillInTheBlankEvaluator

def _run_evaluation_pipeline(config: Dict[str, Any], args: argparse.Namespace):

    # 解析模型列表
    model_names = parse_model_names(args.model) if args.model else [config['llm']['default_model']]
    print(f"📝 评估模型: {', '.join(model_names)}")
    
    # 检查API密钥
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ 未找到OpenRouter API密钥")
        print("💡 请设置环境变量: export OPENROUTER_API_KEY=sk-your-api-key")
        print("🔗 获取密钥: https://openrouter.ai")
        return False
    
    return _run_single_model_evaluation(config, args, model_names[0])


def _run_single_model_evaluation(config, args, model_name):

    print(f"📝 单模型评估: {model_name}")
    
    # 创建LLM配置
    llm_config = create_llm_config(config['llm'], model_name, {})   

    # #common basic
    type='com_bas_fill_in_blank_questions'
    questions = load_jsonl(config['files'][type])
    system_prompt = config.get('fillq_system_prompt')
    evaluator = FillInTheBlankEvaluator(llm_config, system_prompt)    #单选还是主观题的差异体现在这里
    results_file, report_file=generate_output_filenames(model_name,type)
    # 执行评估
    results = execute_evaluation(evaluator, questions, config, results_file)
    # 生成报告
    report = evaluator.generate_report(results, report_file)
    evaluator.print_results(results)
    print_evaluation_completion(results_file, report_file, report)

    #wiring error   scq
    type='bas_wiring_error_ident'
    questions = load_jsonl(config['files'][type])
    system_prompt = config.get('scq_system_prompt')
    evaluator = PCBBenchmarkEvaluator(llm_config, system_prompt)    #单选还是主观题的差异体现在这里
    results_file, report_file=generate_output_filenames(model_name,type)
    # 执行评估
    results = execute_evaluation(evaluator, questions, config, results_file)
    # 生成报告
    report = evaluator.generate_report(results, report_file)
    evaluator.print_results(results)
    print_evaluation_completion(results_file, report_file, report)

    #common basic choice
    type='com_bas_choice_questions'
    questions = load_jsonl(config['files'][type])
    system_prompt = config.get('scq_system_prompt')
    evaluator = PCBBenchmarkEvaluator(llm_config, system_prompt)    #单选还是主观题的差异体现在这里
    results_file, report_file=generate_output_filenames(model_name,type)
    # 执行评估
    results = execute_evaluation(evaluator, questions, config, results_file)
    # 生成报告
    report = evaluator.generate_report(results, report_file)
    evaluator.print_results(results)
    print_evaluation_completion(results_file, report_file, report)

    #Qa placement
    type="qa_pla_questions"
    questions = load_jsonl(config['files'][type])
    system_prompt = config.get('qa_system_prompt')
    evaluator = PCBQAEvaluator(llm_config, system_prompt)
    results_file, report_file=generate_output_filenames(model_name,type)
    # 执行评估
    results = execute_evaluation(evaluator, questions, config, results_file)
    # 生成报告
    report = evaluator.generate_report(results, report_file)
    evaluator.print_results(results)
    print_evaluation_completion(results_file, report_file, report)

    #Qa routing
    type="qa_rou_questions"
    questions = load_jsonl(config['files'][type])
    system_prompt = config.get('qa_system_prompt')
    evaluator = PCBQAEvaluator(llm_config, system_prompt)
    results_file, report_file=generate_output_filenames(model_name,type)
    # 执行评估
    results = execute_evaluation(evaluator, questions, config, results_file)
    # 生成报告
    report = evaluator.generate_report(results, report_file)
    evaluator.print_results(results)
    print_evaluation_completion(results_file, report_file, report)

    return True


def load_config_and_env(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """加载配置文件和环境变量"""
    # 加载环境变量
    env_loaded = load_env_file("config/.env")
    if not env_loaded:
        print("⚠️ 未加载到环境变量文件，请确保 config/.env 存在并包含 OPENROUTER_API_KEY")
    
    # 加载配置文件
    return load_yaml_config(config_path)

def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='PCB Benchmark 评估工具')
    
    parser.add_argument(
        '--model', 
        type=str, 
        help='模型名称，多个模型用逗号分隔 (如: deepseek-chat,gpt-4o)'
    )
    
    parser.add_argument(
        '--max-questions', 
        type=int, 
        help='最大评估问题数量'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='配置文件路径 (默认: config/config.yaml)'
    )
    
    return parser

def main():
    """主函数 - 仅用于测试"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 加载配置
    config = load_config_and_env(args.config)
    
    print("🧪 这是runner.py的测试模式")
    print("📝 实际使用请运行: python eval_scq.py 或 python eval_qa.py")

if __name__ == "__main__":
    main()

 