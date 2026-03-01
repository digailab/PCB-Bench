#!/usr/bin/env python3
"""
评估模块
负责执行PCB知识问答的评测逻辑
"""
import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import traceback
import re
from abc import ABC, abstractmethod

# 设置Hugging Face国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from utils import (
    LLMClient, LLMConfig, AsyncLLMClient,
    format_single_choice_prompt_with_image,
    save_jsonl,
    get_evaluation_summary,
    format_single_choice_prompt,
    extract_choice_from_response,
)

# 导入
from bert_score import score
from sentence_transformers import SentenceTransformer, util

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
        self.system_prompt = system_prompt
    
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
            for i, Question in enumerate(questions):
                task = self._evaluate_single_async(async_client, Question, i)
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
    
    def evaluate_sequential_mode(self, questions: List[Dict[str, Any]], 
                               save_progress: bool = True,
                               progress_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        顺序模式评估入口
        
        Args:
            questions: 问题列表
            save_progress: 是否保存结果
            progress_file: 进度文件路径
            
        Returns:
            评估结果列表
        """
        print(f"🚀 开始顺序评估 {len(questions)} 个问题...")
        start_time = time.time()
        
        results = []
        for i, Question in enumerate(questions):
            try:
                result = self.evaluate_single_question(Question) # 这里调用了子类的 evaluate_single_question
                result['index'] = i # 添加索引以便排序
                results.append(result)
                if self._get_progress_callback():
                    self._get_progress_callback()(i + 1, len(questions), result)
            except Exception as e:
                print(f"❌ 第 {i+1} 题评估失败: {e}")
                results.append(self._create_error_result(Question, str(e), i))
            
            # 保存进度
            if save_progress and progress_file:
                save_jsonl(results, progress_file)
        
        end_time = time.time()
        print(f"✅ 顺序评估完成，耗时 {end_time - start_time:.2f} 秒")
        
        return results

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

    def evaluate_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个问题
        
        Args:
            question_data: 问题数据
            
        Returns:
            评估结果
        """
        question_id = question_data.get('Id', 'unknown')
        correct_answer = question_data.get('Answer')
        

        Question = question_data.get('Question')
        prompt=f"Question: {Question}"
        image_path = question_data.get('ImagePath')
        response = self.llm_client.generate_response_with_image(prompt, image_path, self.system_prompt)

        # 提取预测答案
        prediction = extract_choice_from_response(response) if response else None
        
        # 判断正确性
        is_correct = prediction == correct_answer if prediction and correct_answer else False
        
        result = {
            'question_id': question_id,
            'Question': question_data.get('Question', ''),
            'image_path': image_path if image_path else None,
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
        question_id = question_data.get('Id', 'unknown')
        correct_answer = question_data.get('Answer')
        

        Question = question_data.get('Question')
        prompt=f"Question: {Question}"
        image_path = question_data.get('ImagePath')
        
        try:
            response = await async_client.generate_response_async_with_image(prompt,image_path,self.system_prompt)
            
            # 提取预测答案
            prediction = extract_choice_from_response(response) if response else None
            
            # 判断正确性
            is_correct = prediction == correct_answer if prediction and correct_answer else False
            
            result = {
                'question_id': question_id,
                'Question': question_data.get('Question', ''),
                'image_path': image_path if image_path else None,
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
            'Question': question_data.get('Question', ''),
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
                print(f"\n{i+1}. Id: {case.get('question_id')}")
                print(f"   问题: {case.get('Question', '')[:100]}...")
                print(f"   正确答案: {case.get('correct_answer')}")
                print(f"   预测答案: {case.get('prediction')}")
        
        print("\n" + "="*50)

class PCBQAEvaluator(BaseEvaluator):
    """PCB问答题评估器"""
    
    def __init__(self, llm_config: LLMConfig, system_prompt: Optional[str] = None):
        super().__init__(llm_config, system_prompt)
        # 加载预训练模型
        self.sbert_model = SentenceTransformer('all-mpnet-base-v2')  

    def format_qa_prompt(self, question_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        格式化问答题提示
        """
        Question = question_data.get('Question', '')
        image_path = question_data.get('Image', '')
        
        return f"Question: {Question}", image_path

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

    def evaluate_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个问答题
        
        Args:
            question_data: 问题数据
            
        Returns:
            评估结果
        """
        question_id = question_data.get('Id', 'unknown')
        reference_answer = question_data.get('Answer', '')

        Question = question_data.get('Question')
        prompt=f"Question: {Question}"
        image_path = question_data.get('ImagePath')
        
        response = self.llm_client.generate_response_with_image(prompt, image_path, self.system_prompt)
        
        # 计算F1分数
        f1_scores = self.calculate_f1_score(response or "", reference_answer)
        
        # 计算BERTScore分数
        bertscore_value = 0.0
        if response and reference_answer:
            P, R, F1 = score(cands=[response], refs=[reference_answer], lang="en", verbose=False)
            bertscore_value = F1.mean().item() if F1 is not None else 0.0

        # 计算SBERT分数
        sbert_score_value = 0.0
        if response and reference_answer:
            # 编码参考答案和预测答案
            ref_embedding = self.sbert_model.encode(reference_answer, convert_to_tensor=True)
            pred_embedding = self.sbert_model.encode(response, convert_to_tensor=True)
            # 计算余弦相似度
            sbert_score_value = util.cos_sim(ref_embedding, pred_embedding).item()
        
        result = {
            'question_id': question_id,
            'Question': question_data.get('Question', ''),
            'image_path': image_path if image_path else None,
            'reference_answer': reference_answer,
            'prediction': response,
            'f1_score': f1_scores['f1'],
            'precision': f1_scores['precision'],
            'recall': f1_scores['recall'],
            'bertscore': bertscore_value,
            'sbert_score': sbert_score_value,
            'timestamp': datetime.now().isoformat()
        }
        
        return result

    async def _evaluate_single_async(self, async_client: AsyncLLMClient, 
                                   question_data: Dict[str, Any], 
                                   index: int) -> Dict[str, Any]:
        """
        异步评估单个问答题
        """
        question_id = question_data.get('Id', 'unknown')
        correct_answer = question_data.get('Answer')
        

        Question = question_data.get('Question')
        prompt=f"Question: {Question}"
        image_path = question_data.get('ImagePath')
        
        try:
            response = await async_client.generate_response_async_with_image(prompt, image_path, self.system_prompt)
            # 计算F1分数
            f1_scores = self.calculate_f1_score(response or "", correct_answer)
            
            # 计算BERTScore分数
            bertscore_value = 0.0
            if response and correct_answer:
                P, R, F1 = score(cands=[response], refs=[correct_answer], lang="en", verbose=False)
                bertscore_value = F1.mean().item() if F1 is not None else 0.0

            # 计算SBERT分数
            sbert_score_value = 0.0
            if response and correct_answer:
                # 编码参考答案和预测答案
                ref_embedding = self.sbert_model.encode(correct_answer, convert_to_tensor=True)
                pred_embedding = self.sbert_model.encode(response, convert_to_tensor=True)
                # 计算余弦相似度
                sbert_score_value = util.cos_sim(ref_embedding, pred_embedding).item()
            
            result = {
                'question_id': question_id,
                'Question': question_data.get('Question', ''),
                'image_path': image_path if image_path else None,
                'reference_answer': correct_answer,
                'prediction': response,
                'f1_score': f1_scores['f1'],
                'precision': f1_scores['precision'],
                'recall': f1_scores['recall'],
                'bertscore': bertscore_value,
                'sbert_score': sbert_score_value,
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
            'Question': question_data.get('Question', ''),
            'reference_answer': question_data.get('Answer', ''),
            'prediction': None,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'bleurt_score': 0.0,
            'xcomet_score': 0.0,
            'error': error_msg,
            'timestamp': datetime.now().isoformat(),
            'index': index
        }

    def _get_progress_callback(self) -> callable:
        """获取进度回调函数"""
        def progress_callback(completed: int, total: int, latest_result: Dict[str, Any]):
            progress = completed / total * 100
            f1_score = latest_result.get('f1_score', 0.0)
            print(f"进度: {completed}/{total} ({progress:.1f}%) - F1: {f1_score:.3f} - {latest_result['question_id']}")
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
                'average_f1': 0.0,
                'average_precision': 0.0,
                'average_recall': 0.0,
                'average_bertscore': 0.0,
                'average_sbert': 0.0
            }
        
        avg_f1 = sum(r['f1_score'] for r in valid_results) / len(valid_results)
        avg_precision = sum(r['precision'] for r in valid_results) / len(valid_results)
        avg_recall = sum(r['recall'] for r in valid_results) / len(valid_results)
        avg_bertscore = sum(r.get('bertscore', 0.0) for r in valid_results) / len(valid_results)
        avg_sbert = sum(r.get('sbert_score', 0.0) for r in valid_results) / len(valid_results)
        
        return {
            'total_questions': total,
            'valid_predictions': len(valid_results),
            'average_f1': avg_f1,
            'average_precision': avg_precision,
            'average_recall': avg_recall,
            'average_bertscore': avg_bertscore,
            'average_sbert': avg_sbert
        }

    def _analyze_qa_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        详细分析问答题结果
        """
        analysis = {}
        
        # F1分数分布
        f1_scores = [r.get('f1_score', 0.0) for r in results if r.get('f1_score') is not None]
        if f1_scores:
            analysis['f1_distribution'] = {
                'min': min(f1_scores),
                'max': max(f1_scores),
                'mean': sum(f1_scores) / len(f1_scores),
                'median': sorted(f1_scores)[len(f1_scores) // 2]
            }
        
        # BERTScore分数分布
        bertscore_scores = [r.get('bertscore', 0.0) for r in results if r.get('bertscore') is not None]
        if bertscore_scores:
            analysis['bertscore_distribution'] = {
                'min': min(bertscore_scores),
                'max': max(bertscore_scores),
                'mean': sum(bertscore_scores) / len(bertscore_scores),
                'median': sorted(bertscore_scores)[len(bertscore_scores) // 2]
            }

        # SBERT分数分布
        sbert_scores = [r.get('sbert_score', 0.0) for r in results if r.get('sbert_score') is not None]
        if sbert_scores:
            analysis['sbert_distribution'] = {
                'min': min(sbert_scores),
                'max': max(sbert_scores),
                'mean': sum(sbert_scores) / len(sbert_scores),
                'median': sorted(sbert_scores)[len(sbert_scores) // 2]
            }
        
        # 低分案例（F1 < 0.3）
        low_score_cases = []
        for result in results:
            if result.get('f1_score', 0.0) < 0.3:
                low_score_cases.append({
                    'question_id': result.get('question_id'),
                    'f1_score': result.get('f1_score', 0.0),
                    'Question': result.get('Question', '')[:100] + '...' if len(result.get('Question', '')) > 100 else result.get('Question', '')
                })
        
        analysis['low_score_cases'] = low_score_cases[:10]  # 只保留前10个低分案例
        analysis['total_low_score'] = len(low_score_cases)
        
        return analysis

    def print_results(self, results: List[Dict[str, Any]]):
        """
        打印问答题评估结果
        """
        summary = self._get_qa_evaluation_summary(results)
        
        print("\n" + "="*50)
        print("PCB问答题 评估结果")
        print("="*50)
        print(f"总问题数: {summary['total_questions']}")
        print(f"有效预测: {summary['valid_predictions']}")
        print(f"平均F1分数: {summary['average_f1']:.3f}")
        print(f"平均精确率: {summary['average_precision']:.3f}")
        print(f"平均召回率: {summary['average_recall']:.3f}")
        print(f"平均BERTScore分数: {summary['average_bertscore']:.3f}")
        print(f"平均SBERT分数: {summary['average_sbert']:.3f}")
        print("="*50)
        
        # 显示低分案例
        low_score_cases = [r for r in results if r.get('f1_score', 0.0) < 0.3]
        if low_score_cases:
            print(f"\n低分案例 (F1 < 0.3, 前5个):")
            for i, case in enumerate(low_score_cases[:5]):
                print(f"\n{i+1}. Id: {case.get('question_id')}")
                print(f"   问题: {case.get('Question', '')[:100]}...")
                print(f"   F1分数: {case.get('f1_score', 0.0):.3f}")
        
        print("\n" + "="*50)

class FillInTheBlankEvaluator(BaseEvaluator):

    def __init__(self, llm_config: LLMConfig, system_prompt: Optional[str] = None):
        # Use the specific prompt for fill-in-the-blank questions
        super().__init__(llm_config, system_prompt)
    
    def _extract_numeric_answer(self, response: str) -> Optional[str]:
        """
        Extracts a single numeric answer from the response.
        Assumes the response contains only the numeric answer.
        """
        if not response:
            return None
        # Regex to find a sequence of digits, potentially surrounded by whitespace
        match = re.search(r'^\s*(\d+)\s*$', response)
        if match:
            return match.group(1)
        return None

    def evaluate_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single fill-in-the-blank question.
        """
        question_id = question_data.get('Id', 'unknown')
        correct_answer = str(question_data.get('Answer'))
        
        # Determine prompt and image path
        Question = question_data.get('Question')
        prompt=f"Question: {Question}"
        image_path = question_data.get('ImagePath')
        response = self.llm_client.generate_response_with_image(prompt, image_path, self.system_prompt)
        
        prediction = response
        
        is_correct = prediction == correct_answer if prediction and correct_answer else False
        
        result = {
            'question_id': question_id,
            'Question': question_data.get('Question', ''),
            'image_path': image_path,
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
        Asynchronously evaluates a single fill-in-the-blank question.
        """
        question_id = question_data.get('Id', 'unknown')
        correct_answer = question_data.get('Answer')
        
        try:
            Question = question_data.get('Question')
            prompt=f"Question: {Question}"
            image_path = question_data.get('ImagePath')
            response = await async_client.generate_response_async_with_image(prompt, image_path, self.system_prompt)
            
            prediction = response
            
            is_correct = prediction == correct_answer if prediction and correct_answer else False
            
            result = {
                'question_id': question_id,
                'Question': question_data.get('Question', ''),
                'image_path': image_path,
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
        Creates an error result dictionary.
        """
        return {
            'question_id': question_data.get('id', f'question_{index}'),
            'Question': question_data.get('Question', ''),
            'correct_answer': question_data.get('correct_answer'),
            'prediction': None,
            'raw_response': None,
            'correct': False,
            'error': error_msg,
            'timestamp': datetime.now().isoformat(),
            'index': index
        }
    
    def _get_progress_callback(self) -> callable:
        """
        Returns a progress callback function.
        """
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
        Generates a report with accuracy as the only metric.
        """
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r.get('correct', False))
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        
        report = {
            'evaluation_summary': {
                'total_questions': total_questions,
                'correct_answers': correct_answers,
                'accuracy': accuracy
            },
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'model_name': self.llm_client.config.model_name,
                'temperature': self.llm_client.config.temperature,
                'max_tokens': self.llm_client.config.max_tokens
            }
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report

    def print_results(self, results: List[Dict[str, Any]]):
        """
        Prints the evaluation results, focusing on accuracy.
        """
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r.get('correct', False))
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        
        print("\n" + "="*50)
        print("Fill-in-the-Blank Evaluator Results")
        print("="*50)
        print(f"Total Questions: {total_questions}")
        print(f"Correct Answers: {correct_answers}")
        print(f"Accuracy: {accuracy:.2%}")
        print("="*50)
