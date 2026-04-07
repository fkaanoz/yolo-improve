import os
import yaml


def check_annotation_matching(data_yaml_path):
    # 加载data.yaml配置
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # 获取训练集图像路径和对应标注路径
    train_images_dir = data["train"]
    # 从图像路径推导标注路径（YOLO默认结构：images->labels）
    train_labels_dir = train_images_dir.replace("images", "labels")  # 替换image为labels

    # 检查标注文件夹是否存在
    if not os.path.exists(train_labels_dir):
        print(f"错误：标注文件夹不存在 -> {train_labels_dir}")
        return

    # 获取图像文件列表（只处理常见图像格式）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(train_images_dir)
                   if os.path.splitext(f)[1].lower() in image_extensions]

    if not image_files:
        print(f"警告：图像文件夹中未找到图像文件 -> {train_images_dir}")
        return

    # 随机检查5张图像的标注文件是否存在
    check_count = min(5, len(image_files))
    for i in range(check_count):
        img_name = image_files[i]
        img_basename = os.path.splitext(img_name)[0]  # 去除扩展名
        label_file = f"{img_basename}.txt"
        label_path = os.path.join(train_labels_dir, label_file)

        print(f"图像文件：{img_name}")
        if os.path.exists(label_path):
            # 检查标注文件是否为空
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                print(f"  标注文件存在，内容行数：{len(lines)}")
                print(f"  标注内容示例：{lines[0]}")
            else:
                print(f"  警告：标注文件存在但为空 -> {label_file}")
        else:
            print(f"  错误：未找到对应标注文件 -> {label_file}")
        print("-" * 60)


# 运行检查（替换为你的data.yaml路径）
check_annotation_matching("bvn/data.yaml")
