import os
import shutil
from tqdm import tqdm

def copy_folder_contents(source_folder, target_folder):
    """
    复制一个文件夹中的所有内容到另一个文件夹。
    
    :param source_folder: 源文件夹路径
    :param target_folder: 目标文件夹路径
    """
    # 确保源文件夹存在
    if not os.path.exists(source_folder):
        print(f"源文件夹 {source_folder} 不存在。")
        return
    
    # 如果目标文件夹不存在，创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"目标文件夹 {target_folder} 已创建。")
    
    # 遍历源文件夹中的所有文件和子文件夹
    # for item in os.listdir(source_folder):
    for item in tqdm(os.listdir(source_folder), desc="Processing items", unit="item"):
        source_path = os.path.join(source_folder, item)
        target_path = os.path.join(target_folder, item)
        
        # 如果是文件，直接复制
        if os.path.isfile(source_path):
            shutil.copy2(source_path, target_path)
            # print(f"文件 {item} 已复制")
        # 如果是文件夹，递归复制
        elif os.path.isdir(source_path):
            shutil.copytree(source_path, target_path)
            # print(f"文件夹 {item} 已复制")

if __name__ == "__main__":
    
    # 第5步
    # 目前已经把label部分处理结束,并且根据sparse结果进行空和非空分开.
    # 将semi里面的unlabel image和annotation放入到semisparse里面的unlabel_image和unlabel_annotation里面
    
    
    # 源文件夹路径
    source_folder = "/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/adata/trainval_ss2/10/semi/unlabel_image"
    # 目标文件夹路径
    target_folder = "/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/adata/trainval_ss2/10/semisparse/10/unlabel_image"
    
    # 调用函数复制文件夹内容
    copy_folder_contents(source_folder, target_folder)
    print("复制完成。")