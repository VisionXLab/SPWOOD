import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def separate_labeled_unlabeled_images(img_folder, txt_folder, output_root):
    # 定义输出文件夹路径
    unlabel_img_folder = os.path.join(output_root, "unlabel_image")
    unlabeled_annotation_folder = os.path.join(output_root, "unlabel_annotation")

    # 创建输出文件夹
    os.makedirs(unlabel_img_folder, exist_ok=True)
    os.makedirs(unlabeled_annotation_folder, exist_ok=True)

    # 获取所有图片和标注文件名（不含后缀）
    img_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_folder) if f.endswith(".jpg")}
    txt_files = {os.path.splitext(f)[0]: f for f in os.listdir(txt_folder) if f.endswith(".txt")}
    
    def process_file(img_name, img_file):
        img_path = os.path.join(img_folder, img_file)
        txt_path = os.path.join(txt_folder, txt_files[img_name]) if img_name in txt_files else None

        if txt_path and os.path.exists(txt_path):
            # 检查txt文件是否为空或只有一个空白行
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()  # 读取内容并移除首尾空白字符
            if not content:  # 如果内容为空
                shutil.copy(img_path, os.path.join(unlabel_img_folder, img_file))
                shutil.copy(txt_path, os.path.join(unlabeled_annotation_folder, txt_files[img_name]))
                os.remove(img_path)
                os.remove(txt_path)
        else:
            # 没有对应的txt文件，视为未标注
            print("error!")

    # 使用多线程加速处理
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda item: process_file(*item), img_files.items()), total=len(img_files), desc="Processing files", unit="file"))

    # 统计文件数量
    count_files(img_folder, txt_folder, unlabel_img_folder, unlabeled_annotation_folder)

def count_files(label_img_folder, labeled_annotation_folder, unlabel_img_folder, unlabeled_annotation_folder):
    print("\nFile counts after separation:")
    print(f"Labeled images: {len(os.listdir(label_img_folder))}")
    print(f"Labeled annotations: {len(os.listdir(labeled_annotation_folder))}")
    print(f"Unlabeled images: {len(os.listdir(unlabel_img_folder))}")
    print(f"Unlabeled annotations: {len(os.listdir(unlabeled_annotation_folder))}")


# 第4步
# 将上一步生成sparse里面的label的部分进行空txt和非空txt分开

# 使用示例
img_folder = "/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/adata/DIOR/4/30/semisparse/20/label_image"  # 替换为存放png图片的文件夹路径
txt_folder = "/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/adata/DIOR/4/30/semisparse/20/label_annotation"    # 替换为存放txt文件的文件夹路径
output_root = "/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/adata/DIOR/4/30/semisparse/20"  # 替换为输出文件夹的路径

separate_labeled_unlabeled_images(img_folder, txt_folder, output_root)