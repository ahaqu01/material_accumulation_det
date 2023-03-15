model = dict(
    train_class="0,1,2,3,4,5,6,7,8,9,10,11,12",
    num_classes=13,
    # factor of model depth/width
    depth=0.33,
    width=0.50,
    # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
    act="silu",
)

test_cfg = dict(
    # output image size during evaluation/test
    input_size=(640, 640),
    # confidence threshold during evaluation/test,
    # boxes whose scores are less than test_conf will be filtered
    conf=0.2,
    # nms threshold
    nmsthre=0.65,
    fuse=False,
    fp16=False,
    legacy=False,
    gpu_id="0",
    trt=True,
    trt_workspace=32,
)
