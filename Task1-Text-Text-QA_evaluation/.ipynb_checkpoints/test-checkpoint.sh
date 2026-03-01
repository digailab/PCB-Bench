#!/bin/bash
set -euo pipefail
shopt -s nullglob   # 没有匹配时返回空数组

# 模型列表
models=(
  "opengvlab/internvl3-78b"
)

cfg="config/config.yaml"
qa_dir="data/processed/qa"
out_dir="results/qa"
mkdir -p "$out_dir"

# 收集所有 qa_dir 下的文件
qa_files=("$qa_dir"/*)
if [ ${#qa_files[@]} -eq 0 ]; then
  echo "⚠️  未在 $qa_dir 下找到任何文件"
  exit 1
fi

for m in "${models[@]}"; do
  echo ">>> 当前运行模型: $m"
  # 修改 llm.default_model
  sed -i -E "s|^[[:space:]]*default_model:.*|  default_model: \"$m\"|" "$cfg"

  for f in "${qa_files[@]}"; do
    fname="$(basename "$f")"
    echo "    -> 使用题库文件: $f；文件名"$fname""
    # 修改 data.single_choice_questions_file
  sed -i -E "s|^[[:space:]]*qa_questions_file:.*|  qa_questions_file: \"$fname\"|" "$cfg"

    safe_model=$(echo "$m" | tr '/:' '__')
    safe_file=$(echo "$fname" | tr '/:' '__')
    echo "开始运行eval_qa.py"
    python -u eval_qa.py | tee "$out_dir/${safe_model}__${safe_file}.log"
  done
done
