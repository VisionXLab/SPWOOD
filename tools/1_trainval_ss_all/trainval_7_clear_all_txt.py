import os
from tqdm import tqdm

def clear_txt_files(folder_path):
    """
    清空指定文件夹下所有 .txt 文件的内容。
    
    :param folder_path: 要清空的文件夹路径
    """
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在。")
        return
    
    # 遍历文件夹中的所有文件
    files = os.listdir(folder_path)
    # 使用 tqdm 包裹文件列表，添加进度条
    for filename in tqdm(files, desc="Processing files", unit="file"):
    # for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 检查是否为 .txt 文件
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            # 打开文件并清空内容
            with open(file_path, 'w') as file:
                file.truncate(0)  # 清空文件内容
            # print(f"文件 {filename} 已清空。")

if __name__ == "__main__":
    # 指定要清空的文件夹路径
    folder_path = "/mnt/nas-new/home/zhanggefan/zw/A_datasets/DOTA/DOTA10/DOTA_split_ss/trainval_semisparse/trainval_30p/sparselabelunlabel/10/unlabeled_annotation"
    
    # 调用函数清空所有 .txt 文件
    clear_txt_files(folder_path)
    print("所有 .txt 文件已清空。")