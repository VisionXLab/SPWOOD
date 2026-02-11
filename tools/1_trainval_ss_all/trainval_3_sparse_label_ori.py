# 这个代码是在ori的基础上，进行全局的标注sparse


import os
import random
from collections import defaultdict
from tqdm import tqdm

# 原始文件所在的目录
source_dir \
      = "/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/adata/trainval_ss1/30/trainval_ori/label_annotation"  # 替换为原始文件所在的目录路径
# 新文件保存的目录
target_dir = "/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/adata/trainval_ss1/30/trainval_ori/label_annotation_50"  # 替换为新文件保存的目标目录路径
# 保存比例
save_ratio = 0.50

# 确保目标目录存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 用于统计每个类别的标注数目
category_counts = defaultdict(int)

# 用于存储所有标注数据
all_annotations = []

files = sorted(os.listdir(source_dir))

# 遍历原始目录中的所有 txt 文件
# for filename in sorted(os.listdir(source_dir)):
for filename in tqdm(files, desc="Processing files", unit="file"):
    if filename.endswith(".txt"):
        source_path = os.path.join(source_dir, filename)  # 原始文件路径
        target_path = os.path.join(target_dir, filename)  # 新文件路径

        # 读取原始文件内容
        with open(source_path, 'r') as file:
            lines = file.readlines()

        # 统计每个类别的标注数目并存储所有标注数据
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 9:  # 确保每行数据格式正确
                category = parts[8]  # 类别名称在第9个位置
                category_counts[category] += 1
                all_annotations.append((category, line))  # 类+标注

# 打印每个类别的标注数目
print("\n类别标注数目分类别统计: ")
total_count=0
for category, count in category_counts.items():
    print(f"{category}: {count}")
    total_count = total_count+count
print("\n类别标注数目全类别统计: ")
print(f"全部标注: {total_count}")


# 按类别分组
category_annotations = defaultdict(list)
for category, line in all_annotations:
    category_annotations[category].append(line)


# 每个类别随机保存 save_ratio
for category, lines in category_annotations.items():
    random.seed(42)
    random.shuffle(lines)  # 打乱顺序
    num_to_keep = int(len(lines) * save_ratio)  # 保留 save_ratio 的数据
    category_annotations[category] = lines[:num_to_keep]

# 每个类别随机保存 save_ratio , 将这些内容重新组合
final_annotations = []
for lines in category_annotations.values():
    final_annotations.extend(lines)

# 将处理后的内容写入新文件
for filename in os.listdir(source_dir):
    if filename.endswith(".txt"):
        source_path = os.path.join(source_dir, filename)  # 原始文件路径
        target_path = os.path.join(target_dir, filename)  # 新文件路径

        # 读取原始文件内容
        with open(source_path, 'r') as file:
            lines = file.readlines()

        # 保留的标注数据
        kept_lines = [line for line in lines if line in final_annotations]

        # 将处理后的内容写入新文件
        with open(target_path, 'w') as new_file:
            new_file.writelines(kept_lines)

        print(f"文件已处理并保存到: {target_path}")
