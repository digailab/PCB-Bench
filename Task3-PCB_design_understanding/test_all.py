import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("⏳ Waiting for debugger attach...")
debugpy.wait_for_client() 
"""
问答题评估脚本
"""

import sys
from runner import (
    create_argument_parser, 
    load_config_and_env, 
    _run_evaluation_pipeline
)

def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 加载配置和环境变量
    # print("✅ 已加载环境变量文件: config/.env")
    print("🎯 PCB Benchmark 问答题评估")
    print("🔗 使用 openrouter 统一接口")
    
    config = load_config_and_env(args.config)
    
    # 运行评估
    success = _run_evaluation_pipeline(config, args)
    
    if success:
        print("\n🎉 评估完成!")
    else:
        print("\n❌ 评估失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
