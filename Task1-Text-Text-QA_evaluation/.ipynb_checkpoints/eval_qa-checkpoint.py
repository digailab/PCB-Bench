#!/usr/bin/env python3
"""
问答题评估脚本
支持批量处理多个模型和多个数据文件
"""

import sys
import os
import glob
from pathlib import Path
from runner import (
    create_argument_parser, 
    load_config_and_env, 
    run_qa_evaluation_pipeline,
    run_batch_qa_evaluation
)
def main():
    """主函数"""

    parser = create_argument_parser()
    # 添加批量处理参数
    parser.add_argument(
        '--batch-mode', 
        action='store_true',
        help='启用批量处理模式，处理所有模型和所有QA文件'
    )
    parser.add_argument(
        '--qa-files-dir',
        type=str,
        default='data/processed/qa',
        help='QA文件目录路径 (默认: data/processed/qa)'
    )
    args = parser.parse_args()
    
    # 加载配置和环境变量
    # print("✅ 已加载环境变量文件: config/.env")
    print("🎯 PCB Benchmark 问答题评估")
    print("🔗 使用 openrouter 统一接口")
    
    config = load_config_and_env(args.config)
    
    # 运行评估
    success = run_qa_evaluation_pipeline(config, args)
    
    if success:
        print("\n🎉 问答题评估完成!")
    else:
        print("\n❌ 问答题评估失败!")
        sys.exit(1)

if __name__ == "__main__":
    main() 