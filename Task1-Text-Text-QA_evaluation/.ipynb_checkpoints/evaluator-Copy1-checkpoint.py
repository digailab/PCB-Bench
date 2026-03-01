#!/usr/bin/env python3
"""
评估模块
负责执行PCB知识问答的评测逻辑
"""

import json
import time
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import traceback
import re
from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
import os
# 设置Hugging Face国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from utils import (
    LLMClient, LLMConfig, AsyncLLMClient,
    load_jsonl, save_jsonl,
    extract_choice_from_response,
    format_single_choice_prompt,
    get_evaluation_summary
)

class BaseEvaluator(ABC):
    """评估器基类"""
    
    def __init__(self, llm_config: LLMConfig, system_prompt: Optional[str] = None):
        """
        初始化评估器
        
        Args:
            llm_config: LLM配置
            system_prompt: 系统提示词
        """
        self.llm_config = llm_config
        self.llm_client = LLMClient(llm_config)
        self.system_prompt = system_prompt or self._get_default_system_prompt()
    
    @abstractmethod
    def _get_default_system_prompt(self) -> str:
        """获取默认的系统提示词"""
        pass
    
    @abstractmethod
    def evaluate_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个问题"""
        pass
    
    @abstractmethod
    async def _evaluate_single_async(self, async_client: AsyncLLMClient, 
                                   question_data: Dict[str, Any], 
                                   index: int) -> Dict[str, Any]:
        """异步评估单个问题"""
        pass
    
    @abstractmethod
    def print_results(self, results: List[Dict[str, Any]]):
        """打印评估结果"""
        pass
    
    @abstractmethod
    def generate_report(self, results: List[Dict[str, Any]], 
                       output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成评估报告"""
        pass

    async def evaluate_all_parallel(self, questions: List[Dict[str, Any]], 
                                  progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        完全并行评测所有问题
        
        Args:
            questions: 问题列表
            progress_callback: 进度回调函数
            
        Returns:
            评估结果列表
        """
        print(f"🚀 开始并行评估 {len(questions)} 个问题...")
        start_time = time.time()
        
        async with AsyncLLMClient(self.llm_config) as async_client:
            # 创建所有任务
            tasks = []
            for i, question in enumerate(questions):
                task = self._evaluate_single_async(async_client, question, i)
                tasks.append(task)
            
            # 并行执行所有任务
            if progress_callback:
                # 带进度监控的并行执行
                results = await self._execute_with_progress(tasks, progress_callback)
            else:
                # 简单的并行执行
                results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ 第 {i+1} 题评估失败: {result}")
                processed_results.append(self._create_error_result(questions[i], str(result), i))
            else:
                processed_results.append(result)
        
        end_time = time.time()
        print(f"✅ 并行评估完成，耗时 {end_time - start_time:.2f} 秒")
        
        return processed_results
    
    async def _execute_with_progress(self, tasks: List, progress_callback: callable) -> List:
        """
        带进度监控的任务执行
        """
        total = len(tasks)
        completed = 0
        results = [None] * total
        
        # 创建任务包装器以支持进度回调
        async def task_wrapper(task, index):
            nonlocal completed
            result = await task
            completed += 1
            if progress_callback:
                progress_callback(completed, total, result)
            results[index] = result
            return result
        
        # 包装所有任务
        wrapped_tasks = [task_wrapper(task, i) for i, task in enumerate(tasks)]
        
        # 并行执行
        await asyncio.gather(*wrapped_tasks, return_exceptions=True)
        
        return results
    
    @abstractmethod
    def _create_error_result(self, question_data: Dict[str, Any], 
                           error_msg: str, index: int = 0) -> Dict[str, Any]:
        """创建错误结果记录"""
        pass
    
    def evaluate_parallel_mode(self, questions: List[Dict[str, Any]], 
                             save_progress: bool = True,
                             progress_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        并行模式评估入口
        
        Args:
            questions: 问题列表
            save_progress: 是否保存结果
            progress_file: 进度文件路径
            
        Returns:
            评估结果列表
        """
        # 运行异步评估
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                self.evaluate_all_parallel(questions, self._get_progress_callback())
            )
        finally:
            loop.close()
        
        # 排序结果以保持问题顺序
        results.sort(key=lambda x: x.get('index', 0))
        
        # 保存结果
        if save_progress and progress_file:
            save_jsonl(results, progress_file)
        
        return results
    
    @abstractmethod
    def _get_progress_callback(self) -> callable:
        """获取进度回调函数"""
        pass

class PCBBenchmarkEvaluator(BaseEvaluator):
    """PCB Benchmark评估器"""
        
    def _get_default_system_prompt(self) -> str:
        """获取默认的系统提示词"""
        return """你是一个专业的PCB设计和电子工程专家，具有丰富的电路设计、PCB布局、信号完整性分析等方面的知识。

请根据你的专业知识回答以下PCB相关的单选题。对于每个问题：
1. 仔细分析问题和各个选项
2. 运用你的PCB设计知识进行推理
3. 选择最正确的答案
4. 只需要回答选项字母（A、B、C、D或E），不需要解释

请保持专业和准确。"""

    def evaluate_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个问题
        
        Args:
            question_data: 问题数据
            
        Returns:
            评估结果
        """
        question_id = question_data.get('id', 'unknown')
        correct_answer = question_data.get('correct_answer')
        
        # 格式化提示
        prompt = format_single_choice_prompt(question_data)
        
        # 获取LLM响应
        response = self.llm_client.generate_response(prompt, self.system_prompt)
        
        # 提取预测答案
        prediction = extract_choice_from_response(response) if response else None
        
        # 判断正确性
        is_correct = prediction == correct_answer if prediction and correct_answer else False
        
        result = {
            'question_id': question_id,
            'question': question_data.get('question', ''),
            'correct_answer': correct_answer,
            'prediction': prediction,
            'raw_response': response,
            'correct': is_correct,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    async def _evaluate_single_async(self, async_client: AsyncLLMClient, 
                                   question_data: Dict[str, Any], 
                                   index: int) -> Dict[str, Any]:
        """
        异步评估单个问题
        
        Args:
            async_client: 异步LLM客户端
            question_data: 问题数据
            index: 问题索引
            
        Returns:
            评估结果
        """
        question_id = question_data.get('id', f'question_{index}')
        correct_answer = question_data.get('correct_answer')
        
        try:
            # 格式化提示
            prompt = format_single_choice_prompt(question_data)
            
            # 获取LLM响应
            response = await async_client.generate_response_async(prompt, self.system_prompt)
            
            # 提取预测答案
            prediction = extract_choice_from_response(response) if response else None
            
            # 判断正确性
            is_correct = prediction == correct_answer if prediction and correct_answer else False
            
            result = {
                'question_id': question_id,
                'question': question_data.get('question', ''),
                'correct_answer': correct_answer,
                'prediction': prediction,
                'raw_response': response,
                'correct': is_correct,
                'timestamp': datetime.now().isoformat(),
                'index': index
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 评估问题 {question_id} 失败: {e}")
            return self._create_error_result(question_data, str(e), index)
    
    def _create_error_result(self, question_data: Dict[str, Any], 
                           error_msg: str, index: int = 0) -> Dict[str, Any]:
        """
        创建错误结果记录
        """
        return {
            'question_id': question_data.get('id', f'question_{index}'),
            'question': question_data.get('question', ''),
            'correct_answer': question_data.get('correct_answer'),
            'prediction': None,
            'raw_response': None,
            'correct': False,
            'error': error_msg,
            'timestamp': datetime.now().isoformat(),
            'index': index
        }
    
    def _get_progress_callback(self) -> callable:
        """获取进度回调函数"""
        def progress_callback(completed: int, total: int, latest_result: Dict[str, Any]):
            progress = completed / total * 100
            if latest_result.get('correct'):
                print(f"进度: {completed}/{total} ({progress:.1f}%) - ✓ {latest_result['question_id']}")
            else:
                print(f"进度: {completed}/{total} ({progress:.1f}%) - ✗ {latest_result['question_id']}")
        return progress_callback
    
    def generate_report(self, results: List[Dict[str, Any]], 
                       output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        生成评估报告
        
        Args:
            results: 评估结果列表
            output_file: 输出文件路径
            
        Returns:
            报告数据
        """
        # 获取基本统计信息
        summary = get_evaluation_summary(results)
        
        # 详细分析
        detailed_analysis = self._analyze_results(results)
        
        # 生成报告
        report = {
            'evaluation_summary': summary,
            'detailed_analysis': detailed_analysis,
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(results),
            'model_config': {
                'model_name': self.llm_client.config.model_name,
                'temperature': self.llm_client.config.temperature,
                'max_tokens': self.llm_client.config.max_tokens
            }
        }
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report
    
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        详细分析结果
        
        Args:
            results: 评估结果列表
            
        Returns:
            分析结果
        """
        # 保证所有选项都有统计
        all_options = ['A', 'B', 'C', 'D', 'E']
        
        # 答案分布统计
        answer_distribution = {}
        for option in all_options:
            total = sum(1 for r in results if r.get('correct_answer') == option)
            correct = sum(1 for r in results if r.get('correct_answer') == option and r.get('correct', False))
            accuracy = correct / total if total > 0 else 0.0
            answer_distribution[option] = {
                'total': total,
                'correct': correct,
                'accuracy': accuracy
            }
        
        # 预测分布统计
        prediction_distribution = {}
        for option in all_options:
            prediction_distribution[option] = sum(1 for r in results if r.get('prediction') == option)
        
        # 添加其他统计
        unknown_count = sum(1 for r in results if not r.get('prediction') or r.get('prediction') not in all_options)
        prediction_distribution['Unknown'] = unknown_count
        
        return {
            'answer_distribution': answer_distribution,
            'prediction_distribution': prediction_distribution
        }
    
    def print_results(self, results: List[Dict[str, Any]]):
        """
        打印评估结果
        
        Args:
            results: 评估结果列表
        """
        summary = get_evaluation_summary(results)
        
        print("\n" + "="*50)
        print("PCB Benchmark 评估结果")
        print("="*50)
        print(f"总问题数: {summary['total_questions']}")
        print(f"正确答案数: {summary['correct_answers']}")
        print(f"准确率: {summary['accuracy']:.2%}")
        print(f"无预测: {summary['no_prediction']}")
        print(f"无效预测: {summary['invalid_prediction']}")
        print(f"有效预测: {summary['valid_predictions']}")
        print("="*50)
        
        # 显示错误案例
        error_cases = [r for r in results if not r.get('correct', False)]
        if error_cases:
            print(f"\n错误案例 (前5个):")
            for i, case in enumerate(error_cases[:5]):
                print(f"\n{i+1}. ID: {case.get('question_id')}")
                print(f"   问题: {case.get('question', '')[:100]}...")
                print(f"   正确答案: {case.get('correct_answer')}")
                print(f"   预测答案: {case.get('prediction')}")
        
        print("\n" + "="*50)

class PCBQAEvaluator(BaseEvaluator):
    """PCB问答题评估器"""
    
    def __init__(self, llm_config: LLMConfig, system_prompt: Optional[str] = None):
        """
        初始化PCB问答题评估器
        
        Args:
            llm_config: LLM配置
            system_prompt: 系统提示词
        """
        super().__init__(llm_config, system_prompt)
        # 初始化SBERT模型
        self.sbert_model = None
        self._init_sbert_model()
    
    def _init_sbert_model(self):
        """初始化SBERT模型"""
        try:
            # 使用中文优化的SBERT模型
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ SBERT模型初始化成功")
        except Exception as e:
            print(f"⚠️ SBERT模型初始化失败: {e}")
            self.sbert_model = None
        
    def _get_default_system_prompt(self) -> str:
        """获取默认的系统提示词"""
        return """你是一个专业的PCB设计和电子工程专家，具有丰富的电路设计、PCB布局、信号完整性分析等方面的知识。

请根据你的专业知识回答以下PCB相关的问答题。对于每个问题：
1. 仔细分析问题的要求
2. 运用你的PCB设计知识进行分析
3. 提供准确、详细、专业的答案
4. 答案应该简洁明了，重点突出

请保持专业和准确。"""

    def format_qa_prompt(self, question_data: Dict[str, Any]) -> str:
        """
        格式化问答题提示
        """
        return f"问题：{question_data.get('question', '')}"

    def calculate_f1_score(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        计算F1分数（基于词汇重叠）
        
        Args:
            prediction: 预测答案
            reference: 参考答案
            
        Returns:
            包含precision, recall, f1的字典
        """
        if not prediction or not reference:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # 清理和分词
        def clean_and_tokenize(text):
            # 移除标点符号
            text = re.sub(r'[，。！？、；：""''()（）【】\\[\\]{}《》<>,.!?;:"\'\\-\\s]+', ' ', text)
            # 分割成字符或词汇
            tokens = []
            # 中文按字符分割
            for char in text:
                if char.strip():
                    tokens.append(char)
            return set(tokens)
        
        pred_words = clean_and_tokenize(prediction)
        ref_words = clean_and_tokenize(reference)
        
        if not pred_words and not ref_words:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        if not pred_words:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if not ref_words:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # 计算重叠词汇
        common_words = pred_words.intersection(ref_words)
        
        # 计算precision, recall, f1
        precision = len(common_words) / len(pred_words) if pred_words else 0.0
        recall = len(common_words) / len(ref_words) if ref_words else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def calculate_sbert_similarity(self, prediction: str, reference: str) -> float:
        """
        计算SBERT语义相似度
        
        Args:
            prediction: 预测答案
            reference: 参考答案
            
        Returns:
            SBERT相似度分数 (0-1)
        """
        if not prediction or not reference:
            return 0.0
        
        if self.sbert_model is None:
            return 0.0
        
        try:
            # 获取句子嵌入
            pred_embedding = self.sbert_model.encode([prediction])
            ref_embedding = self.sbert_model.encode([reference])
            
            # 计算余弦相似度
            similarity = np.dot(pred_embedding[0], ref_embedding[0]) / (
                np.linalg.norm(pred_embedding[0]) * np.linalg.norm(ref_embedding[0])
            )
            
            return float(similarity)
        except Exception as e:
            print(f"⚠️ SBERT相似度计算失败: {e}")
            return 0.0
    
    def calculate_bert_score(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        计算BERT-Score
        
        Args:
            prediction: 预测答案
            reference: 参考答案
            
        Returns:
            包含precision, recall, f1的字典
        """
        if not prediction or not reference:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        try:
            # 计算BERT-Score
            P, R, F1 = bert_score([prediction], [reference], lang="zh", verbose=False)
            
            return {
                'precision': float(P[0]),
                'recall': float(R[0]),
                'f1': float(F1[0])
            }
        except Exception as e:
            print(f"⚠️ BERT-Score计算失败: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def evaluate_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个问答题
        
        Args:
            question_data: 问题数据
            
        Returns:
            评估结果
        """
        question_id = question_data.get('id', 'unknown')
        reference_answer = question_data.get('answer', '')
        
        # 格式化提示
        prompt = self.format_qa_prompt(question_data)
        
        # 获取LLM响应
        response = self.llm_client.generate_response(prompt, self.system_prompt)
        
        # 计算各种评估指标
        f1_scores = self.calculate_f1_score(response or "", reference_answer)
        sbert_similarity = self.calculate_sbert_similarity(response or "", reference_answer)
        bert_scores = self.calculate_bert_score(response or "", reference_answer)
        
        result = {
            'question_id': question_id,
            'question': question_data.get('question', ''),
            'reference_answer': reference_answer,
            'prediction': response,
            # 传统F1分数
            'f1_score': f1_scores['f1'],
            'precision': f1_scores['precision'],
            'recall': f1_scores['recall'],
            # SBERT语义相似度
            'sbert_similarity': sbert_similarity,
            # BERT-Score
            'bert_precision': bert_scores['precision'],
            'bert_recall': bert_scores['recall'],
            'bert_f1': bert_scores['f1'],
            'timestamp': datetime.now().isoformat()
        }
        
        return result

    async def _evaluate_single_async(self, async_client: AsyncLLMClient, 
                                   question_data: Dict[str, Any], 
                                   index: int) -> Dict[str, Any]:
        """
        异步评估单个问答题
        """
        question_id = question_data.get('id', f'question_{index}')
        reference_answer = question_data.get('answer', '')
        
        try:
            # 格式化提示
            prompt = self.format_qa_prompt(question_data)
            
            # 获取LLM响应
            response = await async_client.generate_response_async(prompt, self.system_prompt)
            
            # 计算各种评估指标
            f1_scores = self.calculate_f1_score(response or "", reference_answer)
            sbert_similarity = self.calculate_sbert_similarity(response or "", reference_answer)
            bert_scores = self.calculate_bert_score(response or "", reference_answer)
            
            result = {
                'question_id': question_id,
                'question': question_data.get('question', ''),
                'reference_answer': reference_answer,
                'prediction': response,
                # 传统F1分数
                'f1_score': f1_scores['f1'],
                'precision': f1_scores['precision'],
                'recall': f1_scores['recall'],
                # SBERT语义相似度
                'sbert_similarity': sbert_similarity,
                # BERT-Score
                'bert_precision': bert_scores['precision'],
                'bert_recall': bert_scores['recall'],
                'bert_f1': bert_scores['f1'],
                'timestamp': datetime.now().isoformat(),
                'index': index
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 评估问答题 {question_id} 失败: {e}")
            return self._create_error_result(question_data, str(e), index)

    def _create_error_result(self, question_data: Dict[str, Any], 
                           error_msg: str, index: int = 0) -> Dict[str, Any]:
        """创建错误结果记录"""
        return {
            'question_id': question_data.get('id', f'question_{index}'),
            'question': question_data.get('question', ''),
            'reference_answer': question_data.get('answer', ''),
            'prediction': None,
            # 传统F1分数
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            # SBERT语义相似度
            'sbert_similarity': 0.0,
            # BERT-Score
            'bert_precision': 0.0,
            'bert_recall': 0.0,
            'bert_f1': 0.0,
            'error': error_msg,
            'timestamp': datetime.now().isoformat(),
            'index': index
        }

    def _get_progress_callback(self) -> callable:
        """获取进度回调函数"""
        def progress_callback(completed: int, total: int, latest_result: Dict[str, Any]):
            progress = completed / total * 100
            f1_score = latest_result.get('f1_score', 0.0)
            sbert_sim = latest_result.get('sbert_similarity', 0.0)
            bert_f1 = latest_result.get('bert_f1', 0.0)
            print(f"进度: {completed}/{total} ({progress:.1f}%) - F1: {f1_score:.3f} - SBERT: {sbert_sim:.3f} - BERT-F1: {bert_f1:.3f} - {latest_result['question_id']}")
        return progress_callback

    def generate_report(self, results: List[Dict[str, Any]], 
                       output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        生成问答题评估报告
        
        Args:
            results: 评估结果列表
            output_file: 输出文件路径
            
        Returns:
            报告数据
        """
        # 获取基本统计信息
        summary = self._get_qa_evaluation_summary(results)
        
        # 详细分析
        detailed_analysis = self._analyze_qa_results(results)
        
        # 生成报告
        report = {
            'evaluation_summary': summary,
            'detailed_analysis': detailed_analysis,
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(results),
            'model_config': {
                'model_name': self.llm_client.config.model_name,
                'temperature': self.llm_client.config.temperature,
                'max_tokens': self.llm_client.config.max_tokens
            }
        }
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report

    def _get_qa_evaluation_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取问答题评估摘要
        """
        total = len(results)
        valid_results = [r for r in results if r.get('f1_score') is not None]
        
        if not valid_results:
            return {
                'total_questions': total,
                'valid_predictions': 0,
                # 传统F1指标
                'average_f1': 0.0,
                'average_precision': 0.0,
                'average_recall': 0.0,
                # SBERT指标
                'average_sbert_similarity': 0.0,
                # BERT-Score指标
                'average_bert_f1': 0.0,
                'average_bert_precision': 0.0,
                'average_bert_recall': 0.0
            }
        
        # 计算传统F1指标
        avg_f1 = sum(r['f1_score'] for r in valid_results) / len(valid_results)
        avg_precision = sum(r['precision'] for r in valid_results) / len(valid_results)
        avg_recall = sum(r['recall'] for r in valid_results) / len(valid_results)
        
        # 计算SBERT指标
        avg_sbert = sum(r.get('sbert_similarity', 0.0) for r in valid_results) / len(valid_results)
        
        # 计算BERT-Score指标
        avg_bert_f1 = sum(r.get('bert_f1', 0.0) for r in valid_results) / len(valid_results)
        avg_bert_precision = sum(r.get('bert_precision', 0.0) for r in valid_results) / len(valid_results)
        avg_bert_recall = sum(r.get('bert_recall', 0.0) for r in valid_results) / len(valid_results)
        
        return {
            'total_questions': total,
            'valid_predictions': len(valid_results),
            # 传统F1指标
            'average_f1': avg_f1,
            'average_precision': avg_precision,
            'average_recall': avg_recall,
            # SBERT指标
            'average_sbert_similarity': avg_sbert,
            # BERT-Score指标
            'average_bert_f1': avg_bert_f1,
            'average_bert_precision': avg_bert_precision,
            'average_bert_recall': avg_bert_recall
        }

    def _analyze_qa_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        详细分析问答题结果
        """
        analysis = {}
        
        # 传统F1分数分布
        f1_scores = [r.get('f1_score', 0.0) for r in results if r.get('f1_score') is not None]
        if f1_scores:
            analysis['f1_distribution'] = {
                'min': min(f1_scores),
                'max': max(f1_scores),
                'mean': sum(f1_scores) / len(f1_scores),
                'median': sorted(f1_scores)[len(f1_scores) // 2]
            }
        
        # SBERT相似度分布
        sbert_scores = [r.get('sbert_similarity', 0.0) for r in results if r.get('sbert_similarity') is not None]
        if sbert_scores:
            analysis['sbert_distribution'] = {
                'min': min(sbert_scores),
                'max': max(sbert_scores),
                'mean': sum(sbert_scores) / len(sbert_scores),
                'median': sorted(sbert_scores)[len(sbert_scores) // 2]
            }
        
        # BERT-Score F1分布
        bert_f1_scores = [r.get('bert_f1', 0.0) for r in results if r.get('bert_f1') is not None]
        if bert_f1_scores:
            analysis['bert_f1_distribution'] = {
                'min': min(bert_f1_scores),
                'max': max(bert_f1_scores),
                'mean': sum(bert_f1_scores) / len(bert_f1_scores),
                'median': sorted(bert_f1_scores)[len(bert_f1_scores) // 2]
            }
        
        # 低分案例（传统F1 < 0.3）
        low_f1_cases = []
        for result in results:
            if result.get('f1_score', 0.0) < 0.3:
                low_f1_cases.append({
                    'question_id': result.get('question_id'),
                    'f1_score': result.get('f1_score', 0.0),
                    'sbert_similarity': result.get('sbert_similarity', 0.0),
                    'bert_f1': result.get('bert_f1', 0.0),
                    'question': result.get('question', '')[:100] + '...' if len(result.get('question', '')) > 100 else result.get('question', '')
                })
        
        # 低SBERT相似度案例（< 0.5）
        low_sbert_cases = []
        for result in results:
            if result.get('sbert_similarity', 0.0) < 0.5:
                low_sbert_cases.append({
                    'question_id': result.get('question_id'),
                    'f1_score': result.get('f1_score', 0.0),
                    'sbert_similarity': result.get('sbert_similarity', 0.0),
                    'bert_f1': result.get('bert_f1', 0.0),
                    'question': result.get('question', '')[:100] + '...' if len(result.get('question', '')) > 100 else result.get('question', '')
                })
        
        analysis['low_f1_cases'] = low_f1_cases[:10]  # 只保留前10个低F1案例
        analysis['total_low_f1'] = len(low_f1_cases)
        analysis['low_sbert_cases'] = low_sbert_cases[:10]  # 只保留前10个低SBERT案例
        analysis['total_low_sbert'] = len(low_sbert_cases)
        
        return analysis

    def print_results(self, results: List[Dict[str, Any]]):
        """
        打印问答题评估结果
        """
        summary = self._get_qa_evaluation_summary(results)
        
        print("\n" + "="*60)
        print("PCB问答题 评估结果")
        print("="*60)
        print(f"总问题数: {summary['total_questions']}")
        print(f"有效预测: {summary['valid_predictions']}")
        print("\n📊 评估指标:")
        print(f"  传统F1分数: {summary['average_f1']:.3f}")
        print(f"  传统精确率: {summary['average_precision']:.3f}")
        print(f"  传统召回率: {summary['average_recall']:.3f}")
        print(f"  SBERT相似度: {summary['average_sbert_similarity']:.3f}")
        print(f"  BERT-Score F1: {summary['average_bert_f1']:.3f}")
        print(f"  BERT-Score 精确率: {summary['average_bert_precision']:.3f}")
        print(f"  BERT-Score 召回率: {summary['average_bert_recall']:.3f}")
        print("="*60)
        
        # 显示低分案例
        low_f1_cases = [r for r in results if r.get('f1_score', 0.0) < 0.3]
        if low_f1_cases:
            print(f"\n🔍 低F1分数案例 (F1 < 0.3, 前5个):")
            for i, case in enumerate(low_f1_cases[:5]):
                print(f"\n{i+1}. ID: {case.get('question_id')}")
                print(f"   问题: {case.get('question', '')[:100]}...")
                print(f"   传统F1: {case.get('f1_score', 0.0):.3f}")
                print(f"   SBERT: {case.get('sbert_similarity', 0.0):.3f}")
                print(f"   BERT-F1: {case.get('bert_f1', 0.0):.3f}")
        
        # 显示低SBERT相似度案例
        low_sbert_cases = [r for r in results if r.get('sbert_similarity', 0.0) < 0.5]
        if low_sbert_cases:
            print(f"\n🔍 低SBERT相似度案例 (相似度 < 0.5, 前5个):")
            for i, case in enumerate(low_sbert_cases[:5]):
                print(f"\n{i+1}. ID: {case.get('question_id')}")
                print(f"   问题: {case.get('question', '')[:100]}...")
                print(f"   传统F1: {case.get('f1_score', 0.0):.3f}")
                print(f"   SBERT: {case.get('sbert_similarity', 0.0):.3f}")
                print(f"   BERT-F1: {case.get('bert_f1', 0.0):.3f}")
        
        print("\n" + "="*60) 