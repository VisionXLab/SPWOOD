# 依据json文件进行半监督的划分，分别是标注部分和未标注部分

import os
import shutil
import json
from tqdm import tqdm

# 读取 JSON 文件
json_file_path = './adata/30/0.3.json'
with open(json_file_path, 'r') as json_file:
    selected_file_names = json.load(json_file)

# 源文件夹路径
source_folder_png = './trainval/images'  # 包含 .png 文件的文件夹
source_folder_txt = './trainval/annfiles'  # 包含 .txt 文件的文件夹

# 目标文件夹路径
target_folder_png_selected = './adata/30/trainval_ori/label_image'  # 目标文件夹，用于存放选中的 .png 文件
target_folder_txt_selected = './adata/30/trainval_ori/label_annotation'  # 目标文件夹，用于存放选中的 .txt 文件
target_folder_png_unselected = './adata/30/trainval_ori/unlabel_image'  # 目标文件夹，用于存放未选中的 .png 文件
target_folder_txt_unselected = './adata/30/trainval_ori/unlabel_annotation'  # 目标文件夹，用于存放未选中的 .txt 文件

# 确保目标文件夹存在
os.makedirs(target_folder_png_selected, exist_ok=True)
os.makedirs(target_folder_txt_selected, exist_ok=True)
os.makedirs(target_folder_png_unselected, exist_ok=True)
os.makedirs(target_folder_txt_unselected, exist_ok=True)

# 获取所有文件名
all_png_files = {f for f in os.listdir(source_folder_png) if f.endswith('.jpg')}
all_txt_files = {f for f in os.listdir(source_folder_txt) if f.endswith('.txt')}

# 遍历所有文件名，将选中的文件复制到目标文件夹，未选中的文件复制到另一个目标文件夹
# for png_file in all_png_files:
for png_file in tqdm(all_png_files, desc="Processing PNG files", unit="file"):
    # 提取文件名的基本部分（例如 P0000）
    base_name = png_file.split('.')[0]
    
    # 对应的 .txt 文件名
    txt_file = png_file.replace('.jpg', '.txt')
    
    if base_name in selected_file_names:
        # 选中的文件
        shutil.copy(os.path.join(source_folder_png, png_file), target_folder_png_selected)
        shutil.copy(os.path.join(source_folder_txt, txt_file), target_folder_txt_selected)
    else:
        # 未选中的文件
        shutil.copy(os.path.join(source_folder_png, png_file), target_folder_png_unselected)
        shutil.copy(os.path.join(source_folder_txt, txt_file), target_folder_txt_unselected)

print("文件筛选和复制完成。")