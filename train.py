import sys
import argparse
import os
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 指定显卡和多卡训练问题 统一都在<YOLOV11配置文件.md>下方常见错误和解决方案。
# 训练过程中loss出现nan，可以尝试关闭AMP，就是把下方amp=False的注释去掉。
# 训练时候输出的AMP Check使用的YOLO11n的权重不是代表载入了预训练权重的意思，只是用于测试AMP，正常的不需要理会。
#C2DA=0.721,C2CGA=0.73 ,yolo11n=0.727,yolo11-C3k2-AdditiveBlock=0.686，yolo11-CA-HSFPN=0.702
# ,yolo11-CAFMFusion=0.728,yolo11-CGAFusion=0.745,yolo11-FeaturePyramidSharedConv=0.729
# yolo11-LSDECD=0.709,yolo11-GLSA=0.719,yolo11-RSCD=0.600
if __name__ == '__main__':
    model = YOLO('YOLO-SW.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='bvn/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0',·
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                #amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                seed=0,  # 随机种子
                deterministic=True,  # 是否使用确定性操作
                single_cls=False,  # 是否只针对单个类进行训练
                rect=False,  # 是否使用矩形锚框
                cos_lr=False,  # 是否使用余弦学习率衰减
                warmup_epochs=8.0,  # 预热轮数
                warmup_momentum=0.8,  # 预热动量
                warmup_bias_lr=0.2,  # 预热偏差学习率
                save_json=False,  # 是否保存验证结果为 JSON
                save_hybrid=False,  # 是否使用混合保存模式
                max_det=300,  # 每张图像最大检测目标数量
                conf=0.001,  # 置信度阈值
                #iou=0.7,  # IoU阈值
                plots=True,  # 是否生成训练过程的图形化可视化
                nbs=64,  # 基于样本尺寸的批量大小
                lr0=0.002,  # 初始学习率
                lrf=0.05,  # 最终学习率
                momentum=0.937,  # 动量
                weight_decay=0.0005,  # 权重衰减
                # 其他参数...
                )