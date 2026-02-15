# 从trainval构建json文件。

import os
import random
import json

random.seed(42)  # 设置随机种子为 42

def select_random_images(folder_path, ratio, output_json):
    """
    从指定文件夹中随机选取指定比例的图片文件名，并保存到 JSON 文件中。

    :param folder_path: 包含图片的文件夹路径
    :param ratio: 选取的图片比例（0 到 1 之间）
    :param output_json: 保存结果的 JSON 文件路径
    """
    # 获取文件夹中所有 .png 文件
    all_images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    # 计算需要选取的图片数量
    num_to_select = int(len(all_images) * ratio)
    
    # 随机选取指定数量的图片
    selected_images = random.sample(all_images, num_to_select)
    
    # 提取图片文件名（不包含扩展名）
    selected_image_names = [os.path.splitext(img)[0] for img in selected_images]
    
    # 保存到 JSON 文件
    with open(output_json, 'w') as json_file:
        json.dump(selected_image_names, json_file, indent=4)
    
    print(f"已随机选取 {num_to_select} 张图片，结果已保存到 {output_json}")

if __name__ == "__main__":
    # 指定文件夹路径
    folder_path = "./trainval"
    
    # 指定选取比例
    ratio = 0.3  # 例如，选取 30% 的图片
    
    # 指定输出 JSON 文件路径
    output_json = "./0.3.json"
    
    # 调用函数
    select_random_images(folder_path, ratio, output_json)