import pandas as pd
import json
import os

def read_xlsx_and_convert_to_json(xlsx_file_path, output_dir_scq,output_dir_qa,File_name):
    """
    读取xlsx文件并转换为两个JSON格式文件
    
    Args:
        xlsx_file_path: xlsx文件路径
        output_dir: 输出目录
    """
    
    # 创建输出目录
    if not os.path.exists(output_dir_scq):
        os.makedirs(output_dir_scq)
    if not os.path.exists(output_dir_qa):
        os.makedirs(output_dir_qa)
    
    # 读取xlsx文件
    try:
        df = pd.read_excel(xlsx_file_path)
        print(f"成功读取xlsx文件，共{len(df)}行数据")
        print(f"列名: {list(df.columns)}")
    except Exception as e:
        print(f"读取xlsx文件失败: {e}")
        return
    
    # 初始化两个JSON数据列表
    qa_data = []
    scq_data = []
    
    # 遍历每一行数据
    for index, row in df.iterrows():
        try:
            # 获取数据（注意pandas的列索引从0开始）
            # 根据描述：question在第6列(索引5)，answer在第7列(索引6)，choices在第8列(索引7)，correct_answer在第5列(索引4)
            question_qa = str(row.iloc[6]) if pd.notna(row.iloc[6]) else ""  # 第6列
            question_scq = str(row.iloc[7]) if pd.notna(row.iloc[7]) else ""
            answer = str(row.iloc[8]) if pd.notna(row.iloc[8]) else ""    # 第7列
            choices = str(row.iloc[9]) if pd.notna(row.iloc[9]) else ""   # 第8列
            correct_answer = str(row.iloc[5]) if pd.notna(row.iloc[5]) else ""  # 第5列
            
            # 跳过空行
            if not question_qa.strip():
                continue
            
            # 生成ID
            qa_id = f"qa_{index}"
            scq_id = f"scq_{index}"
            
            # 创建QA格式数据
            qa_item = {
                "id": qa_id,
                "question": question_qa,
                "answer": answer
            }
            qa_data.append(qa_item)
            
            # 解析选项（假设选项格式为 "A.选项1 B.选项2 C.选项3 D.选项4 E.选项5"）
            options = []
            if choices.strip():
                # 按选项分割（A、B、C、D、E）
                choice_parts = choices.split()
                current_option = ""
                option_letter = ""
                
                for part in choice_parts:
                    if part.startswith(('A.', 'B.', 'C.', 'D.', 'E.')):
                        # 如果之前有选项，先保存
                        if current_option and option_letter:
                            options.append([option_letter, current_option.strip()])
                        # 开始新选项
                        option_letter = part[0]  # A, B, C, D, E
                        current_option = part[2:]  # 去掉"A."等前缀
                    else:
                        # 继续当前选项的内容
                        current_option += " " + part
                
                # 保存最后一个选项
                if current_option and option_letter:
                    options.append([option_letter, current_option.strip()])
            
            # 如果没有解析到选项，创建默认选项
            if not options:
                options = [
                    ["A", "选项A"],
                    ["B", "选项B"],
                    ["C", "选项C"],
                    ["D", "选项D"],
                    ["E", "以上选项都不正确"]
                ]
            
            # 创建SCQ格式数据
            scq_item = {
                "id": scq_id,
                "question": question_scq,
                "options": options,
                "correct_answer": correct_answer
            }
            scq_data.append(scq_item)
            
        except Exception as e:
            print(f"处理第{index}行数据时出错: {e}")
            continue
    
    # 保存QA格式JSON文件
    qa_output_path = os.path.join(output_dir_qa, f"{File_name}_qa.jsonl")
    with open(qa_output_path, 'w', encoding='utf-8') as f:
        for item in qa_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"QA格式文件已保存到: {qa_output_path}")
    
    # 保存SCQ格式JSON文件
    scq_output_path = os.path.join(output_dir_scq, f"{File_name}_scq.jsonl")
    with open(scq_output_path, 'w', encoding='utf-8') as f:
        for item in scq_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"SCQ格式文件已保存到: {scq_output_path}")
    
    print(f"转换完成！共处理{len(qa_data)}条QA数据和{len(scq_data)}条SCQ数据")

def main():
    # 设置输入和输出目录
    input_dir = "data/raw"
    output_dir_scq = "data/processed/scq"
    output_dir_qa = "data/processed/qa"

    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误：找不到输入目录 {input_dir}")
        return
    
    # 获取目录中所有的xlsx文件
    xlsx_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]
    
    if not xlsx_files:
        print(f"在目录 {input_dir} 中没有找到xlsx文件")
        return
    
    print(f"找到 {len(xlsx_files)} 个xlsx文件，开始处理...")
    
    # 遍历所有xlsx文件
    for xlsx_file in xlsx_files:
        # 获取文件名（不包含扩展名）
        file_name = os.path.splitext(xlsx_file)[0]
        xlsx_file_path = os.path.join(input_dir, xlsx_file)
        
        print(f"\n正在处理文件: {xlsx_file}")
        
        # 执行转换
        read_xlsx_and_convert_to_json(xlsx_file_path, output_dir_scq,output_dir_qa, file_name)

    
    print(f"\n所有文件处理完成！共处理了 {len(xlsx_files)} 个文件")

if __name__ == "__main__":
    main()
