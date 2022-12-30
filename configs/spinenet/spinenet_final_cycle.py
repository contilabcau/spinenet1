# Model_structure 
# https://mmdetection.readthedocs.io/en/latest/tutorials/config.html
# Config System 
# https://mmdetection.readthedocs.io/en/v2.5.0/config.html

cudnn_benchmark = True # Accelerate training when input size is fixed
# Refer to https://github.com/open-mmlab/mmdetection/issues/725

# model settings
norm_cfg = dict(type='SyncBN', momentum=0.01, eps=1e-3, requires_grad=True) # https://github.com/yan-roo/SpineNet-Pytorch/blob/master/configs/spinenet/mask_rcnn_spinenet_49_B_8gpu_640.py


model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='SpineNet',
        arch="49",
        norm_cfg=norm_cfg),
    neck=None,
    rpn_head=dict(
        type='RPNHead', # The type of RPN head is 'RPNHead', we also support 'GARPNHead', etc. 
        # Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/rpn_head.py#L12 for more details.
        in_channels=256, # The input channels of each input feature map, this is consistent with the output channels of neck
        feat_channels=256, # Feature channels of convolutional layers in the head.
        anchor_generator=dict( # The config of anchor generator
            type='AnchorGenerator', # Most of methods use AnchorGenerator, SSD Detectors uses `SSDAnchorGenerator'
            # Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py#L10 for more details
            scales=[3], # Basic scale of the anchor, the area of the anchor in one position of a feature map will be scale * base_sizes
            ratios=[0.5, 1.0, 2.0], # The ratio between height and width.
            strides=[8, 16, 32, 64, 128]), # The strides of the anchor generator.
            # This is consistent with the FPN feature strides. The strides will be taken as base_sizes if base_sizes is not set.
        bbox_coder=dict( # Config of box coder to encode and decode the boxes during training and testing
            type='DeltaXYWHBBoxCoder', # Type of box coder. 'DeltaXYWHBBoxCoder' is applied for most of methods.
            # Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L9 for more details.
            target_means=[.0, .0, .0, .0], # The target means used to encode and decode boxes
            target_stds=[1.0, 1.0, 1.0, 1.0]), # The standard variance used to encode and decode
        loss_cls=dict( # Config of loss function for the classification branch
            type='CrossEntropyLoss', # Type of loss for classification branch, we also support 
            use_sigmoid=True, # RPN usually perform two-class classification, so it usually uses sigmoid function.
            loss_weight=1.0), # Loss weight of the classification branch.
        loss_bbox=dict( # Config of loss function for the regression branch.
            type='SmoothL1Loss', # Type of loss, we also support many IoU Losses and smooth L1-loss, etc. 
            # Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/smooth_l1_loss.py#L56 for implementation.
            beta=1.0 / 9.0, 
            loss_weight=1.0)), # Loss weight of the regression branch.
    roi_head=dict( # RoIHead encapsulates the second stage of two-stage/cascade detectors.
        type='StandardRoIHead', # Type of the RoI head. 
        # Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/standard_roi_head.py#L10 for implementation.
        bbox_roi_extractor=dict( # RoI feature extractor for bbox regression.
            type='SingleRoIExtractor', # Type of the RoI feature extractor, most of methods uses SingleRoIExtractor. 
            # Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/roi_extractors/single_level.py#L10 for details.
            roi_layer=dict( # Config of RoI Layer
                type='RoIAlign', # Type of RoI Layer, DeformRoIPoolingPack and ModulatedDeformRoIPoolingPack are also supported. 
                # Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/roi_align/roi_align.py#L79 for details.
                output_size=7, # The output size of feature maps.
                sampling_ratio=0), # Sampling ratio when extracting the RoI features. 0 means adaptive ratio.          
            out_channels=256, # output channels of the extracted feature.
            featmap_strides=[8, 16, 32, 64]), # Strides of multi-scale feature maps. 
            # It should be consistent to the architecture of the backbone.
        bbox_head=dict( # Config of box head in the RoIHead.
            type='ConvFCBBoxHead', # Type of the bbox head, 
            # Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L11 for implementation details.
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256, # Input channels for bbox head.
            # This is consistent with the out_channels in roi_extractor
            conv_out_channels=256,
            fc_out_channels=1024, # Output feature channels of FC layers.
            roi_feat_size=7, # Size of RoI features
            num_classes=178, # Number of classes for classification
            bbox_coder=dict( # Box coder used in the second stage.
                type='DeltaXYWHBBoxCoder', # Type of box coder.
                # 'DeltaXYWHBBoxCoder' is applied for most of methods.
                clip_border=True, # https://mmdetection.readthedocs.io/en/latest/_modules/mmdet/core/bbox/coder/tblr_bbox_coder.html
                # Whether clip the objects outside the border of the image. Defaults to True.
                target_means=[0., 0., 0., 0.], # Means used to encode and decode box
                target_stds=[0.1, 0.1, 0.2, 0.2]), # Standard variance for encoding and decoding.
                # It is smaller since the boxes are more accurate. [0.1, 0.1, 0.2, 0.2] is a conventional setting.
            reg_class_agnostic=False, # Whether the regression is class agnostic.
            # Refer to https://arxiv.org/pdf/2011.14204.pdf
            # The proposed class-agnostic detection aims to localize all objects irrespective of their types including those of unknown classes.
            norm_cfg=norm_cfg,
            loss_cls=dict( # Config of loss function for the classification branch
                type='CrossEntropyLoss', # Type of loss for classification branch, we also support FocalLoss etc.
                use_sigmoid=False, # Whether to use sigmoid.
                loss_weight=1.0), # Loss weight of the classification branch.
            loss_bbox=dict( # Config of loss function for the regression branch.
                type='SmoothL1Loss', # Type of loss, we also support many IoU Losses and smooth L1-loss, etc.
                beta=1.0, 
                loss_weight=1.0)), # Loss weight of the regression branch.
        mask_roi_extractor=dict( # RoI feature extractor for bbox regression.
            type='SingleRoIExtractor', # Type of the RoI feature extractor, most of methods uses SingleRoIExtractor.
            roi_layer=dict( # Config of RoI Layer that extracts features for instance segmentation
                type='RoIAlign', # Type of RoI Layer, DeformRoIPoolingPack and ModulatedDeformRoIPoolingPack are also supported
                out_size=14, # The output size of feature maps.
                sample_num=2),
            out_channels=256, # Output channels of the extracted feature.
            featmap_strides=[8, 16, 32, 64]), # Strides of multi-scale feature maps.
        mask_head=dict( # Mask prediction head
            type='FCNMaskHead',  # Type of mask head, 
            # Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py#L21 for implementation details.
            num_convs=4, # Number of convolutional layers in mask head.
            in_channels=256, # Input channels, should be consistent with the output channels of mask roi extractor.
            conv_out_channels=256, # Output channels of the convolutional layer.
            num_classes=178, # Number of class to be segmented.
            norm_cfg=norm_cfg,
            loss_mask=dict( # Config of loss function for the mask branch.
                type='CrossEntropyLoss', # Type of loss used for segmentation
                use_mask=True, # Whether to only train the mask in the correct class.
                loss_weight=1.0))), # Loss weight of mask branch.

    # model training and testing settings
    train_cfg = dict( # Config of training hyperparameters for rpn and rcnn
        rpn=dict( # Training config of rpn
            assigner=dict( # Config of assigner
                type='MaxIoUAssigner', # Type of assigner, MaxIoUAssigner is used for many common detectors. 
                # Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10 for more details.
                pos_iou_thr=0.7, # IoU >= threshold 0.7 will be taken as positive samples
                neg_iou_thr=0.3, # IoU < threshold 0.3 will be taken as negative samples
                min_pos_iou=0.3, # The minimal IoU threshold to take boxes as positive samples
                match_low_quality=True, # Whether to match the boxes under low quality (see API doc for more details).
                ignore_iof_thr=-1), # IoF threshold for ignoring bboxes
            sampler=dict( # Config of positive/negative sampler
                type='RandomSampler', # Type of sampler, PseudoSampler and other samplers are also supported. 
                # Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8 for implementation details.
                num=256, # Number of samples
                pos_fraction=0.5, # The ratio of positive samples in the total samples.
                neg_pos_ub=-1, # The upper bound of negative samples based on the number of positive samples.
                add_gt_as_proposals=False), # Whether add GT as proposals after sampling.
            allowed_border=0, # The border allowed after padding for valid anchors.
            pos_weight=-1, # The weight of positive samples during training.
            debug=False), # Whether to set the debug mode
        rpn_proposal=dict( # The config to generate proposals during training
            nms_pre=2000, # The number of boxes before NMS
            max_per_img=1000, # The number of boxes to be used after NMS
            nms=dict(
                type='nms', 
                iou_threshold=0.7), # The threshold to be used during NMS
            min_bbox_size=0), # The allowed minimal box size
        rcnn=dict( # The config for the roi heads.
            assigner=dict( # Config of assigner for second stage, this is different for that in rpn
                type='MaxIoUAssigner', # Type of assigner, MaxIoUAssigner is used for all roi_heads for now. 
                # Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10 for more details.
                pos_iou_thr=0.5, # IoU >= threshold 0.5 will be taken as positive samples
                neg_iou_thr=0.5, # IoU >= threshold 0.5 will be taken as positive samples
                min_pos_iou=0.5, # The minimal IoU threshold to take boxes as positive samples
                match_low_quality=False, # Whether to match the boxes under low quality (see API doc for more details).
                ignore_iof_thr=-1), # IoF threshold for ignoring bboxes
            sampler=dict(
                type='RandomSampler', # Type of sampler, PseudoSampler and other samplers are also supported. 
                # Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8 for implementation details.
                num=512, # Number of samples
                pos_fraction=0.25, # The ratio of positive samples in the total samples.
                neg_pos_ub=-1, # The upper bound of negative samples based on the number of positive samples.
                add_gt_as_proposals=True), # Whether add GT as proposals after sampling.
            mask_size=28, # Size of mask
            pos_weight=-1, # The weight of positive samples during training.
            debug=False)), # Whether to set the debug mode
    test_cfg = dict( # Config for testing hyperparameters for rpn and rcnn
        rpn=dict( # The config to generate proposals during testing
            nms_across_levels=False, # Whether to do NMS for boxes across levels
            nms_pre=1000, # The number of boxes before NMS
            max_per_img=1000, # The number of boxes to be used after NMS
            nms=dict(
                type='nms', 
                iou_threshold=0.7), # The threshold to be used during NMS
            min_bbox_size=0), # The allowed minimal box size
        rcnn=dict( # The config for the roi heads.
            score_thr=0.05, # Threshold to filter out boxes
            nms=dict( # Config of nms in the second stage
                type='nms', # Type of nms
                iou_thr=0.5), # NMS threshold
            max_per_img=100,  # Max number of detections of each image
            mask_thr_binary=0.5)) # Threshold of mask prediction
)
# dataset settings
dataset_type = 'CocoDataset' # Dataset type, this will be used to define the dataset
data_root = 'data/final_cycle/' # Root path of data
img_norm_cfg = dict( # Image normalization config to normalize the input images
    mean=[123.675, 116.28, 103.53], # Mean values used to pre-training the pre-trained backbone models
    std=[58.395, 57.12, 57.375], # Standard variance used to pre-training the pre-trained backbone models
    to_rgb=True) # The channel orders of image used to pre-training the pre-trained backbone models
train_pipeline = [ # Training pipeline
    dict(type='LoadImageFromFile'), # First pipeline to load images from file path
    dict(
        type='LoadAnnotations',  # Second pipeline to load annotations for current image
        with_bbox=True, # Whether to use bounding box, True for detection
        with_mask=True, # Whether to use instance mask, True for instance segmentation
        poly2mask=False), # Whether to convert the polygon mask to instance mask, set False for acceleration and to save memory
    dict(
        type='Resize', # Augmentation pipeline that resize the images and their annotations
        img_scale=(640, 640), # The largest scale of image
        ratio_range=(0.5, 2.0),
        keep_ratio=True), # whether to keep the ratio between height and width.
    dict(
        type='RandomCrop', # The absolute `crop_size` is sampled based on `crop_type` and `image_size`, 
        # Then the cropped results are generated. 
        crop_size=(640, 640)), # The relative ratio or absolute pixels of height and width.
        # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/pipelines/transforms.py
    dict(
        type='RandomFlip', # Augmentation pipeline that flip the images and their annotations
        flip_ratio=0.5), # The ratio or probability to flip
    dict(
        type='Normalize', # Augmentation pipeline that normalize the input images
        **img_norm_cfg),
    dict(
        type='Pad', # Padding config
        size=(640, 640)),
    dict(type='DefaultFormatBundle'), # Default format bundle to gather data in the pipeline
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'), # First pipeline to load images from file path
    dict(
        type='MultiScaleFlipAug', # An encapsulation that encapsulates the testing augmentations
        img_scale=(640, 640), # Decides the largest scale for testing, used for the Resize pipeline
        flip=False, # Whether to flip images during testing
        transforms=[
            dict(
                type='Resize', # Use resize augmentation
                keep_ratio=True), # Whether to keep the ratio between height and width 
                # The img_scale set here will be supressed by the img_scale set above.
            dict(type='RandomFlip'), # Thought RandomFlip is added in pipeline, it is not used because flip=False
            dict(
                type='Normalize', # Normalization config, the values are from img_norm_cfg
                **img_norm_cfg),
            dict(
                type='Pad', # Padding config to pad images divisable by 32.
                size_divisor=128), # https://github.com/yan-roo/SpineNet-Pytorch/blob/master/configs/spinenet/mask_rcnn_spinenet_49_B_8gpu_640.py
            dict(
                type='ImageToTensor', # convert image to tensor
                keys=['img']),
            dict(
                type='Collect', # Collect pipeline that collect necessary keys for testing.
                keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4, # Batch size of a single GPU
    workers_per_gpu=2, # Worker to pre-fetch data for each single GPU
    train=dict( # Train dataset config
        type=dataset_type, # Type of dataset, 
        # Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/coco.py#L19 for details.
        ann_file=data_root + 'annotations/ContilTrain.json', # Path of annotation file
        img_prefix=data_root + 'training/', # Prefix of image path
        pipeline=train_pipeline), # pipeline, this is passed by the train_pipeline created before.
    val=dict( # Validation dataset config
        type=dataset_type,
        ann_file=data_root + 'annotations/ContilValidation.json',
        img_prefix=data_root + 'validation/',
        pipeline=test_pipeline),
    test=dict( # Test dataset config, modify the ann_file for test-dev/test submission
        type=dataset_type,
        ann_file=data_root + 'annotations/ContilTest.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))
evaluation = dict(interval=100, metric=['bbox', 'segm'], classwise = True)
# optimizer
optimizer = dict( # The config to build the evaluation hook, 
# Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7 for more details.
    type='SGD', # Type of optimizers, 
    # Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/optimizer/default_constructor.py#L13 for more details
    lr=0.14, # Learning rate of optimizers, see detail usages of the parameters in the documentaion of PyTorch
    momentum=0.9, # he momentum used for updating ema parameter.
    # Ema's parameter are updated with the formula:
    # `ema_param = (1-momentum) * ema_param + momentum * cur_param`. Defaults to 0.0002.
    # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/hook/ema.py
    weight_decay=4e-5) # Weight decay of SGD
optimizer_config = dict( # Config used to build the optimizer hook, 
# Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details.
    grad_clip=None) # Most of the methods do not use gradient clip
# learning policy
lr_config = dict( # Learning rate scheduler config used to register LrUpdater hook
    policy='step', # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. 
    # Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    warmup='linear', # The warmup policy, also support `exp` and `constant`.
    warmup_iters=140, # The number of iterations for warmup
    warmup_ratio=0.1, # The ratio of the starting learning rate used for warmup
    step=[320, 340]) # Steps to decay the learning rate
checkpoint_config = dict( # Config to set the checkpoint hook, 
# Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    interval=1) # The save interval is 1
# yapf:disable
log_config = dict( # config to register logger hook
    interval=10, # Interval to print the log
    hooks=[ # dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported
        dict(type='TextLoggerHook'), # The logger used to record the training process.
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
runner = dict(
    type='EpochBasedRunner',
    max_epochs=100) # Total epochs to train the model
dist_params = dict(backend='nccl') # Parameters to setup distributed training, the port can also be set.
log_level = 'INFO' # The level of logging.
work_dir = 'work_dirs/final_cycle/spinenet' # Save path of pth_file
load_from = None # load models as a pre-trained model from a given path. This will not resume training.
resume_from = 'work_dirs/final_cycle/spinenet/latest.pth'
# resume_from = '/media/contil/My Book Duo/2022_NIA_project/work_dirs/20221201_2_cycle/spinenet/latest.pth' # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1), ('val', 1)] # https://mmdetection.readthedocs.io/en/latest/tutorials/customize_runtime.html
# Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. 
# The workflow trains the model by 350 epochs according to the total_epochs.

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

auto_scale_lr = dict(enable=True, base_batch_size=4) # https://github.com/open-mmlab/mmdetection/pull/7482 # 기본 LR 사용하지 않는 방법

# Note : This is for automatically scaling LR, USER CAN'T CHANGE THIS VALUE
mmdet_official_special_batch_size = 4 # https://github.com/open-mmlab/mmdetection/pull/7377

# default_gpu_number = 4 # https://github.com/open-mmlab/mmdetection/pull/7482/commits/8f950ebf0b2ee8dfa3d741e40c8ed7a2624babfa