#!/usr/bin/env python3
"""
评估运行器模块
负责协调评估流程的执行
"""

import argparse
import os
import sys
import glob
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from utils import (
    load_yaml_config,
    load_env_file,
    create_llm_config,
    create_llm_configs_for_models,
    load_evaluation_questions,
    load_qa_questions,
    generate_output_filenames,
    execute_evaluation,
    execute_sequential_evaluation,
    print_evaluation_completion,
    parse_model_names
)
from evaluator import PCBBenchmarkEvaluator, PCBQAEvaluator

def _run_evaluation_pipeline(config: Dict[str, Any], args: argparse.Namespace, eval_type: str = 'scq'):
    """通用评估流程"""
    eval_name = "单选题" if eval_type == 'scq' else "问答题"
    print(f"🚀 开始{eval_name}评估...")

    # 解析模型列表
    model_names = parse_model_names(args.model) if args.model else [config['llm']['default_model']]
    print(f"📝 评估模型: {', '.join(model_names)}")
    
    # 检查API密钥
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ 未找到OpenRouter API密钥")
        print("💡 请设置环境变量: export OPENROUTER_API_KEY=sk-your-api-key")
        print("🔗 获取密钥: https://openrouter.ai")
        return False
    
    # 执行评估
    if len(model_names) == 1:
        # print(f"📝 单模型{eval_name}评估: {model_names[0]}")
        return _run_single_model_evaluation(config, args, model_names[0], eval_type)
    else:
        # print(f"📝 多模型{eval_name}评估: {', '.join(model_names)}")
        return _run_multi_model_evaluation(config, args, model_names, eval_type)

def _run_single_model_evaluation(config, args, model_name, eval_type='scq'):
    """通用单模型评估"""
    eval_name = "单选题" if eval_type == 'scq' else "问答题"
    print(f"📝 单模型{eval_name}评估: {model_name}")
    
    # 创建LLM配置
    llm_config = create_llm_config(config['llm'], model_name, {})
    
    # 加载数据和创建评估器
    max_questions = config['evaluation'].get('max_questions') or args.max_questions
    if eval_type == 'scq':
        questions = load_evaluation_questions(config, max_questions)
        system_prompt = config.get('system_prompt')
        evaluator = PCBBenchmarkEvaluator(llm_config, system_prompt)
        suffix = f"_{model_name}"
    else:  # qa
        questions = load_qa_questions(config, max_questions)
        system_prompt = config.get('qa_system_prompt')
        evaluation_config = config.get('evaluation', {})
        evaluator = PCBQAEvaluator(llm_config, system_prompt, evaluation_config)
        suffix = f"_qa_{model_name}"
    
    # 生成输出文件名
    results_file, report_file = generate_output_filenames(config, suffix, eval_type)
    
    # 执行评估
    results = execute_evaluation(evaluator, questions, config, results_file)
    
    # 生成报告
    report = evaluator.generate_report(results, report_file)
    evaluator.print_results(results)
    print_evaluation_completion(results_file, report_file, report)
    return True

def _run_multi_model_evaluation(config, args, model_names, eval_type='scq'):
    """通用多模型评估"""
    eval_name = "单选题" if eval_type == 'scq' else "问答题"
    print(f"📝 多模型{eval_name}评估: {len(model_names)} 个模型")
    
    success_count = 0
    
    for i, model_name in enumerate(model_names, 1):
        print(f"\n--- 模型 {i}/{len(model_names)}: {model_name} ---")
        
        try:
            _run_single_model_evaluation(config, args, model_name, eval_type)
            success_count += 1
            print(f"✅ {model_name} 评估完成")
        except Exception as e:
            print(f"❌ {model_name} 评估失败: {e}")
    
    # 总结
    print(f"\n🎉 多模型{eval_name}评估完成!")
    print(f"📊 成功: {success_count}/{len(model_names)}")
    
    return success_count > 0

def run_scq_evaluation_pipeline(config: Dict[str, Any], args: argparse.Namespace) -> bool:
    """单选题评估流程"""
    return _run_evaluation_pipeline(config, args, 'scq')

def run_qa_evaluation_pipeline(config: Dict[str, Any], args: argparse.Namespace) -> bool:
    """问答题评估流程"""
    return _run_evaluation_pipeline(config, args, 'qa')

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

def run_batch_qa_evaluation(config: Dict[str, Any], args: argparse.Namespace) -> bool:
    """
    批量QA评估 - 处理所有模型和所有QA文件
    优化版本：避免重复初始化模型
    """
    print("🚀 开始批量QA评估...")
    
    # 获取所有模型
    if args.model:
        model_names = parse_model_names(args.model)
    else:
        # 使用默认模型列表
        model_names = [
            "openai/gpt-5",
            "openai/gpt-4o", 
            "anthropic/claude-opus-4.1",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.0-flash-exp",
            "deepseek/deepseek-chat",
            "meta-llama/llama-3.1-405b-instruct"
        ]
    
    print(f"📝 评估模型: {', '.join(model_names)}")
    
    # 获取所有QA文件
    qa_files_dir = Path(args.qa_files_dir)
    if not qa_files_dir.exists():
        print(f"❌ QA文件目录不存在: {qa_files_dir}")
        return False
    
    qa_files = list(qa_files_dir.glob("*.jsonl"))
    if not qa_files:
        print(f"❌ 在 {qa_files_dir} 中未找到QA文件")
        return False
    
    print(f"📁 找到 {len(qa_files)} 个QA文件")
    
    # 检查API密钥
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ 未找到OpenRouter API密钥")
        print("💡 请设置环境变量: export OPENROUTER_API_KEY=sk-your-api-key")
        return False
    
    # 批量处理
    total_combinations = len(model_names) * len(qa_files)
    print(f"🎯 总共需要处理 {total_combinations} 个模型-文件组合")
    
    success_count = 0
    start_time = time.time()
    
    for model_idx, model_name in enumerate(model_names, 1):
        print(f"\n{'='*60}")
        print(f"🤖 处理模型 {model_idx}/{len(model_names)}: {model_name}")
        print(f"{'='*60}")
        
        # 为每个模型创建一次评估器（避免重复初始化）
        llm_config = create_llm_config(config['llm'], model_name, {})
        system_prompt = config.get('qa_system_prompt')
        evaluation_config = config.get('evaluation', {})
        evaluator = PCBQAEvaluator(llm_config, system_prompt, evaluation_config)
        
        for file_idx, qa_file in enumerate(qa_files, 1):
            print(f"\n📄 处理文件 {file_idx}/{len(qa_files)}: {qa_file.name}")
            
            # 在try块外定义original_file，确保finally块能访问到
            original_file = None
            
            try:
                # 临时修改配置以使用当前文件
                original_file = config['data'].get('qa_questions_file')
                config['data']['qa_questions_file'] = qa_file.name
                
                # 加载当前文件的问题
                max_questions = config['evaluation'].get('max_questions') or args.max_questions
                questions = load_qa_questions(config, max_questions)
                
                if not questions:
                    print(f"⚠️ 文件 {qa_file.name} 中没有有效问题，跳过")
                    continue
                
                # 生成输出文件名
                safe_model = model_name.replace('/', '_').replace(':', '_')
                safe_file = qa_file.stem
                suffix = f"_qa_{safe_model}_{safe_file}"
                results_file, report_file = generate_output_filenames(config, suffix, 'qa')
                
                # 执行评估
                print(f"🔍 开始评估 {len(questions)} 个问题...")
                results = execute_evaluation(evaluator, questions, config, results_file)
                
                # 生成报告
                report = evaluator.generate_report(results, report_file)
                evaluator.print_results(results)
                print_evaluation_completion(results_file, report_file, report)
                
                success_count += 1
                print(f"✅ {model_name} + {qa_file.name} 评估完成")
                
            except Exception as e:
                print(f"❌ {model_name} + {qa_file.name} 评估失败: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                # 恢复原始配置
                if original_file is not None:
                    config['data']['qa_questions_file'] = original_file
    
    # 总结
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"🎉 批量QA评估完成!")
    print(f"📊 成功处理: {success_count}/{total_combinations} 个组合")
    print(f"⏱️ 总耗时: {total_time:.2f} 秒")
    print(f"⚡ 平均每个组合: {total_time/total_combinations:.2f} 秒")
    print(f"{'='*60}")
    
    return success_count > 0

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

 