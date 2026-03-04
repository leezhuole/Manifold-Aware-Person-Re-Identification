# Protocol-2
CUDA_VISIBLE_DEVICES=0 python examples/train_bau.py -a resnet50 -ds market1501 msmt17 cuhksysu -dt cuhk03 -b 64 --lam 1.5 --k 10 --iters 500 --logs-dir "/home/stud/leez/reid/src/BAU/examples/logs/hyp_m-ms-cs_c3" --manifold-aware
# CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py -a resnet50 -ds market1501 cuhksysu cuhk03 -dt msmt17 -b 256 --lam 1.5 --k 10 --iters 200 --logs-dir "/home/stud/leez/reid/src/BAU/examples/logs/m-cs-c3_ms"
# CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py -a resnet50 -ds msmt17 cuhk03 cuhksysu -dt market1501 -b 256 --lam 1.5 --k 10 --iters 500 --logs-dir "/home/stud/leez/reid/src/BAU/examples/logs/ms-cs-c3_m"

# Protocol-3
# CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py -a resnet50 -ds market1501dg msmt17dg cuhksysu -dt cuhk03 -b 256 --lam 1.5 --k 10 --iters 1000
# CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py -a resnet50 -ds market1501dg cuhksysu cuhk03dg -dt msmt17 -b 256 --lam 1.5 --k 10 --iters 400
# CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py -a resnet50 -ds msmt17dg cuhk03dg cuhksysu -dt market1501 -b 256 --lam 1.5 --k 10 --iters 1000

# Protocol-1
# CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py -a resnet50 -ds market1501dg cuhk02dg cuhk03dg cuhksysu -dt grid -b 256 --lam 1.5 --k 10 --iters 500


CUDA_VISIBLE_DEVICES=0 python examples/train_bau.py \
  -a resnet50_finsler \
  -ds agreidv2_aerial agreidv2_wearable agreidv2_cctv \
  -dt agreidv2_aerial_to_cctv \
  -b  64 \
  --lam 1.5 --k 10 --iters 500 \
  --logs-dir "/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/logs/learnableOmega" \
  --eval-epochs 1 3 5 7 9 \
  --verbosity 2 \
  --wandb-name "learnableOmegaTest_driftdim_128" \
  --epochs 20 \
  --eval-drift false \
  --memory-bank-mode full \
  --drift-dim 128

