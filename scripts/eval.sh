# Target: Market1501
CUDA_VISIBLE_DEVICES=0 python examples/test.py \
    -d market1501 \
    --resume "/home/stud/leez/reid/src/BAU/examples/logs/ms-cs-c3_m/best.pth" \

# Target: CUHK-SYSU
CUDA_VISIBLE_DEVICES=0 python examples/test.py \
    -d cuhk03 \
    --resume "/home/stud/leez/reid/src/BAU/examples/logs/m-ms-cs_c3/best.pth" \

# Target: MSMT17
CUDA_VISIBLE_DEVICES=0 python examples/test.py \
    -d msmt17 \
    --resume "/home/stud/leez/reid/src/BAU/examples/logs/m-cs-c3_ms/best.pth" \