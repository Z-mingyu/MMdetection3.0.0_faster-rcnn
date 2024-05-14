_base_ = [
    '/home/zmy/workspace/mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py',
    '/home/zmy/workspace/mmdetection/configs/_base_/datasets/coco_detection.py',
    '/home/zmy/workspace/mmdetection/configs/_base_/schedules/schedule_1x.py',
    '/home/zmy/workspace/mmdetection/configs/_base_/default_runtime.py'
]

backend_args = None
dataset_type = 'CocoDataset'
classes = ('apple', 'orange', 'medicine box', 'water', 'thermometer',)
data_root = '/home/zmy/workspace/mmdetection/data/coco/'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # 将类别名字添加至 `metainfo` 字段中
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/')
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=False,
        # 将类别名字添加至 `metainfo` 字段中
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/')
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/zmy/workspace/mmdetection/data/coco/annotations/instances_val2017.json',
    metric=['bbox'],
    format_only=False,
    backend_args=backend_args)

model = dict(
    backbone=dict(init_cfg=None),
    roi_head=dict(
        bbox_head=dict(
            num_classes=5,
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))

# optimizer
# lr is set for a batch size of 8
optim_wrapper = dict(optimizer=dict(lr=0.01))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=8,
        by_epoch=True,
        # [7] yields higher performance than [6]
        milestones=[7],
        gamma=0.1)
]

# actual epoch = 8 * 8 = 64
train_cfg = dict(max_epochs=8)

# For better, more stable performance initialize from COCO
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (1 samples per GPU)
# TODO: support auto scaling lr
# auto_scale_lr = dict(base_batch_size=8)
