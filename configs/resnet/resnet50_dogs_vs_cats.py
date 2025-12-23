# configs/resnet/resnet50_dogs_vs_cats.py

# 继承官方 ResNet50 + ImageNet 配置
_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

# ===================== 1. 模型：改成 2 类 =====================
# 只在原始 resnet50 的基础上做“微调用的改动”
model = dict(
    # backbone：加载 ImageNet 预训练，并且冻结前 2 个 stage
    backbone=dict(
        frozen_stages=2,  # 0: 不冻；1: 冻 stem+layer1；2: 冻到 layer2，以此类推
        init_cfg=dict(
            type='Pretrained',
            checkpoint=(
                'https://download.openmmlab.com/mmclassification/'
                'v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
            ),
            prefix='backbone',  # 只把 backbone 部分权重加载进来
        ),
    ),
    # head：把分类数改成 2 类（猫 / 狗）
    head=dict(
        num_classes=2,
    ),
)


# data_preprocessor 里也同步设成 2 类
data_preprocessor = dict(
    num_classes=2,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# ===================== 2. 数据集：用你划分好的 dogs_vs_cats_split =====================
# 假设你按之前脚本，把数据划分到了：
# /root/autodl-tmp/mmpretrain/mmpretrain/dogs_vs_cats_split/
#   train/cat, train/dog
#   val/cat,   val/dog
#   test/cat,  test/dog
# 如果你输出到了别的路径，就把下面的 data_root 改成对应路径

dataset_type = 'CustomDataset'
data_root = '/root/autodl-tmp/mmpretrain/mmpretrain/dogs_vs_cats_split'
classes = ('cat', 'dog')

# 训练 / 测试 pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=64,      # 4090 可以先用 64，不够再往上加
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='',          # 子文件夹格式 -> ann_file 置空
        data_prefix='train',  # -> .../dogs_vs_cats_split/train
        with_label=True,
        metainfo=dict(classes=classes),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='',
        data_prefix='val',    # -> .../dogs_vs_cats_split/val
        with_label=True,
        metainfo=dict(classes=classes),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='',
        data_prefix='test',   # -> .../dogs_vs_cats_split/test
        with_label=True,
        metainfo=dict(classes=classes),
        pipeline=test_pipeline,
    ),
)

# ===================== 3. 训练超参（微调用小一点 LR） =====================
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.01,          # 微调建议小一点
        momentum=0.9,
        weight_decay=0.0001,
    )
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=20,       # 先跑 20 个 epoch，基本够了
    val_interval=1,
)

# 评估指标：简单用 Accuracy
val_evaluator = dict(type='Accuracy', topk=(1,))
test_evaluator = dict(type='Accuracy', topk=(1,))
