import cv2
import numpy as np
import os

# --- 配置您的文件夹路径 (请替换为您的实际路径) ---
IMAGE_DIR = '/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/adata/DIOR/4_semi_sparse/30/semisparse/20/label_image' 
ORIGINAL_TXT_DIR = '/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/adata/DIOR/3_split_ss_dota/trainval/annfiles'
SPARSE_TXT_DIR = '/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/adata/DIOR/4_semi_sparse/30/semisparse/20/label_annotation'
OUTPUT_DIR = '/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/adata/DIOR/4_semi_sparse/30/semisparse/20/GT_sparsGT_image'

# --- 颜色配置 (BGR 顺序，因为 OpenCV 默认使用 BGR) ---
COLOR_SPARSE = (255, 0, 0)      # 蓝色 (Blue) - 用于稀疏标注
COLOR_REMAINING = (0, 0, 255)   # 红色 (Red) - 用于原始标注中剩余的部分
LINE_THICKNESS = 2

# --- 辅助函数：读取 DOTA 格式的 TXT 标注 (已修改以包含类别) ---
def load_dota_annotations(txt_path):
    """
    读取 DOTA 格式的 TXT 文件，返回一个包含 (坐标列表, 类别字符串) 的列表。
    TXT 行格式: x1 y1 x2 y2 x3 y3 x4 y4 category score
    """
    boxes_with_category = []
    if not os.path.exists(txt_path):
        return boxes_with_category
    
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            # 至少需要 8 个坐标和 1 个类别 (共 9 个元素)
            if len(parts) < 9: 
                continue 
            
            try:
                coords = [float(p) for p in parts[:8]]
                category = parts[8] # 类别名称是第 9 个元素
                boxes_with_category.append((coords, category))
            except ValueError:
                print(f"Warning: Skipping malformed line in {txt_path}: {line}")
                continue
    return boxes_with_category

# --- 辅助函数：绘制旋转矩形框 ---
def draw_rotated_box(img, coords, color, thickness):
    """
    在图像上绘制一个由 8 个坐标定义的旋转矩形框。
    """
    if len(coords) != 8:
        return

    pts = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

# --- 辅助函数：绘制类别文字 ---
def draw_text(img, text, coords, color):
    """
    在标注框的左上角位置绘制类别名称。
    """
    # 确定文本位置 (使用第一个坐标点 x1, y1)
    # 将浮点数转换为整数
    x1, y1 = int(coords[0]), int(coords[1])
    
    # 文本参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7  # 略微增大字体
    text_thickness = 2
    
    # 调整文本位置，使其位于框的上方
    text_pos = (x1, y1 - 5) 
    
    # 绘制文字
    cv2.putText(img, text, text_pos, font, font_scale, color, text_thickness, cv2.LINE_AA)


# --- 主程序 ---
def visualize_annotations():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 遍历图片文件夹中的所有 PNG 文件
    for filename in os.listdir(IMAGE_DIR):
        if not filename.endswith('.png'):
            continue
            
        img_id, _ = os.path.splitext(filename)
        img_path = os.path.join(IMAGE_DIR, filename)
        
        # 对应的 TXT 文件路径
        original_txt_path = os.path.join(ORIGINAL_TXT_DIR, img_id + '.txt')
        sparse_txt_path = os.path.join(SPARSE_TXT_DIR, img_id + '.txt')

        # 1. 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            continue

        # 2. 加载标注
        # 格式: [(coords, category), ...]
        original_boxes = load_dota_annotations(original_txt_path)
        sparse_boxes = load_dota_annotations(sparse_txt_path)
        
        # --- 3. 绘制逻辑 ---
        # 策略：先绘制所有原始标注 (红色)，然后用稀疏标注 (蓝色) 覆盖
        # 这样未被稀疏选中的标注将以红色显示，稀疏标注以蓝色显示。
        
        # 绘制所有原始标注 (红色)
        for coords, category in original_boxes:
            draw_rotated_box(img, coords, COLOR_REMAINING, LINE_THICKNESS)
            draw_text(img, category, coords, COLOR_REMAINING)
             
        # 再次绘制稀疏标注 (蓝色) 以覆盖，确保蓝色在上方
        for coords, category in sparse_boxes:
            draw_rotated_box(img, coords, COLOR_SPARSE, LINE_THICKNESS)
            draw_text(img, category, coords, COLOR_SPARSE)

        # 4. 保存结果
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, img)
        print(f"Processed and saved: {output_path}")

# --- 运行可视化 ---
if __name__ == '__main__':
    # !!! 运行前请务必替换 IMAGE_DIR, ORIGINAL_TXT_DIR, SPARSE_TXT_DIR, OUTPUT_DIR 四个路径变量 !!!
    visualize_annotations()