norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    # pretrained=None,
    # UNET网络结构
    # backbone=dict(
    #     type='UNet',
    #     in_channels=3,
    #     base_channels=64,
    #     num_stages=5,
    #     strides=(1, 1, 1, 1, 1),
    #     enc_num_convs=(2, 2, 2, 2, 2),
    #     dec_num_convs=(2, 2, 2, 2),
    #     downsamples=(True, True, True, True),
    #     enc_dilations=(1, 1, 1, 1, 1),
    #     dec_dilations=(1, 1, 1, 1),
    #     with_cp=False,
    #     conv_cfg=None,
    #     norm_cfg=norm_cfg,
    #     act_cfg=dict(type='ReLU'),
    #     upsample_cfg=dict(type='InterpConv'),
    #     norm_eval=False),
    # neck=dict(  # 添加neck
    #     type='FPN',
    #     in_channels=[1024, 512, 256, 128, 64],
    #     out_channels=64,#所有阶段的输出通道数都变为64
    #     num_outs=5),

    # 主干网络修改为VisionTransformer
    # 问题：主干网络修改了，预训练参数是如何加载进来的？
    backbone=dict(
        type='VisionTransformer',
        img_size=(96, 96),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(2, 3, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        interpolate_mode='bicubic'),
    neck=dict(
            type='FPN',
            in_channels=[768, 768, 768, 768, 768],#输入通道数调整为transformer的隐藏层维度
            out_channels=64,
            num_outs=5),

    decode_head=dict(
        type='FCNHead',
        in_channels=64,  # 输入的通道数
        in_index=4,  # 输入的特征层索引
        channels=64,  # Channels after modules, before conv_seg
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=8,  # 类别个数
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=64,  # 输入的通道数
        in_index=3,  # 输入的特征层索引
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(64, 64), stride=(42, 42)))
load_from = "../checkpoints/fcn_unet_s5-d16_64x64_40k_drive_20201223_191051-5daf6d3b.pth"
dataset_type = 'ICCV09Dataset'
data_root = '../data/iccv09Data/'
img_dir = 'images'
ann_dir = 'labels'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=40000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir=img_dir,
            ann_dir=ann_dir,
            split='splits/train.txt',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='Resize',
                    img_scale=(584, 565),
                    ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop', crop_size=(64, 64), cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(64, 64), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ])),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        split='splits/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(584, 565),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        split='splits/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(584, 565),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mDice', pre_eval=True)
work_dir = './work_dirs/fcn_unet_s5-d16_64x64_40k_drive_iccv09'
gpu_ids = [0]
auto_resume = False
