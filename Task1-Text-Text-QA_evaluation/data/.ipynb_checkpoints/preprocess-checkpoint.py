#!/usr/bin/env python3
"""
数据预处理脚本
将xlsx文件中的数据转换为jsonl格式，同时生成单选题和问答题数据
"""

import pandas as pd
import json
import re
import os
from pathlib import Path

def load_xlsx_data(xlsx_path):
    """
    加载xlsx文件数据
    """
    try:
        # 读取xlsx文件
        df = pd.read_excel(xlsx_path, sheet_name=0)
        print(f"成功读取xlsx文件，共 {len(df)} 行数据")
        print(f"列名: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"读取xlsx文件时出错: {e}")
        return None

def parse_qa_question(question_text, answer_text):
    """
    解析问答题，提取问题和答案
    """
    if pd.isna(question_text) or pd.isna(answer_text):
        return None
    
    question_text = str(question_text).strip()
    answer_text = str(answer_text).strip()
    
    if not question_text or not answer_text:
        return None
    
    return {
        'question': question_text,
        'answer': answer_text
    }

def parse_single_choice_question(question_text, answer_text):
    """
    解析单选题，提取问题和选项
    """
    if pd.isna(question_text) or pd.isna(answer_text):
        return None
    
    question_text = str(question_text).strip()
    answer_text = str(answer_text).strip()
    
    # 分割问题和选项
    lines = question_text.split('\n')
    
    # 第一行通常是问题
    question = lines[0].strip()
    
    # 提取选项
    options = []
    
    # 常见的选项格式模式
    patterns = [
        r'([A-E])\.\s*([^\n\r]*)',  # A. 选项内容
        r'([A-E])\)\s*([^\n\r]*)',  # A) 选项内容
        r'([A-E])\s*[:：]\s*([^\n\r]*)',  # A: 选项内容
    ]
    
    for line in lines[1:]:  # 从第二行开始查找选项
        line = line.strip()
        if not line:
            continue
            
        for pattern in patterns:
            matches = re.findall(pattern, line)
            if matches:
                for match in matches:
                    option_letter = match[0].upper()
                    option_text = match[1].strip()
                    if option_text:  # 只添加非空选项
                        options.append((option_letter, option_text))
                break
    
    # 如果没有找到选项，尝试从整个文本中提取
    if not options:
        full_text = question_text
        for pattern in patterns:
            matches = re.findall(pattern, full_text, re.MULTILINE)
            if matches:
                for match in matches:
                    option_letter = match[0].upper()
                    option_text = match[1].strip()
                    if option_text and option_text != "以上选项都不正确":
                        options.append((option_letter, option_text))
    
    # 去除重复选项
    unique_options = []
    seen = set()
    for letter, text in options:
        if letter not in seen:
            unique_options.append((letter, text))
            seen.add(letter)
    
    # 提取正确答案
    correct_answer = None
    if answer_text:
        # 直接从答案文本中提取字母
        answer_match = re.search(r'([A-E])', answer_text.upper())
        if answer_match:
            correct_answer = answer_match.group(1)
    
    return {
        'question': question,
        'options': unique_options,
        'correct_answer': correct_answer
    }

def convert_qa_to_jsonl(df, output_path):
    """
    将DataFrame转换为问答题jsonl格式
    """
    questions = []
    
    if len(df.columns) >= 3:
        question_col = df.columns[1]  # 第2列：问答题问题
        answer_col = df.columns[2]    # 第3列：问答题答案
    else:
        print("数据文件列数不足，无法处理问答题")
        return []
    
    print(f"处理问答题数据，问题列: {question_col}, 答案列: {answer_col}")
    
    for idx, row in df.iterrows():
        question_text = row[question_col]
        answer_text = row[answer_col]
        
        parsed = parse_qa_question(question_text, answer_text)
        if parsed:
            question_data = {
                'id': f"qa_{idx}",
                'question': parsed['question'],
                'answer': parsed['answer']
            }
            questions.append(question_data)
    
    # 写入jsonl文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + '\n')
    
    print(f"成功转换 {len(questions)} 道问答题到 {output_path}")
    return questions

def convert_scq_to_jsonl(df, output_path):
    """
    将DataFrame转换为单选题jsonl格式
    """
    questions = []
    
    # 根据检查结果，选择题数据在第3列（带选项的完整问题）和第4列（答案）
    if len(df.columns) >= 5:
        question_col = df.columns[3]  # 第4列：带选项的完整问题
        answer_col = df.columns[5]    # 第5列：答案选项
    else:
        print("数据文件列数不足，无法处理单选题")
        return []
    
    print(f"处理单选题数据，问题列: {question_col}, 答案列: {answer_col}")
    
    for idx, row in df.iterrows():
        question_text = row[question_col]
        answer_text = row[answer_col]
        
        parsed = parse_single_choice_question(question_text, answer_text)
        if parsed:
            question_data = {
                'id': f"scq_{idx}",
                'question': parsed['question'],
                'options': parsed['options'],
                'correct_answer': parsed['correct_answer']
            }
            questions.append(question_data)
    
    # 写入jsonl文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + '\n')
    
    print(f"成功转换 {len(questions)} 道单选题到 {output_path}")
    return questions

def main():
    """
    主函数
    """
    # 设置路径
    current_dir = Path(__file__).parent
    raw_dir = current_dir / "raw"
    processed_dir_qa = current_dir / "processed_finetune/qa"
    processed_dir_scq = current_dir / "processed_finetune/scq"
    
    # 创建processed目录
    processed_dir_qa.mkdir(exist_ok=True)
    processed_dir_scq.mkdir(exist_ok=True)
    
    # 找到xlsx文件
    xlsx_files = list(raw_dir.glob("*.xlsx"))
    if not xlsx_files:
        print("在data/raw目录中未找到xlsx文件")
        return
    
    # 依次处理raw目录下的每个xlsx数据文件
    for xlsx_path in xlsx_files:
        print(f"\n============================")
        print(f"处理文件: {xlsx_path}")
        print(f"============================")
        
        # 加载数据
        df = load_xlsx_data(xlsx_path)
        if df is None:
            continue
        
        # 显示数据预览
        print("\n数据预览:")
        print(df.head())
        print(f"\n数据形状: {df.shape}")
        
        # 使用原始数据文件名作为输出文件名的一部分
        base_name = xlsx_path.stem
        
        # 转换问答题：processed_finetune/qa/原始文件名_qa.jsonl
        qa_output_path = processed_dir_qa / f"{base_name}_qa.jsonl"
        qa_questions = convert_qa_to_jsonl(df, qa_output_path)
        
        # 转换单选题：processed_finetune/scq/原始文件名_scq.jsonl
        scq_output_path = processed_dir_scq / f"{base_name}_scq.jsonl"
        scq_questions = convert_scq_to_jsonl(df, scq_output_path)
        
        # 显示示例
        if qa_questions:
            print("\n问答题转换示例:")
            for i, q in enumerate(qa_questions[:2]):
                print(f"\n问答题 {i+1}:")
                print(json.dumps(q, ensure_ascii=False, indent=2))
        
        if scq_questions:
            print("\n单选题转换示例:")
            for i, q in enumerate(scq_questions[:2]):
                print(f"\n单选题 {i+1}:")
                print(json.dumps(q, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main() 