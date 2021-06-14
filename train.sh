env OMP_NUM_THREADS=4 python3 -m torch.distributed.launch --nproc_per_node="$1" train.py
