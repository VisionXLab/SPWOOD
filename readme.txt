cd /mnt/nas-new/home/zhanggefan/zw/mcl
conda activate zw_mcl

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --master_port=25520 \
train.py /mnt/nas-new/home/zhanggefan/zw/mcl/configs_dota15/mcl/30p.py --launcher pytorch --work-dir zz_result/20_full/


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --master_port=25550 \
train.py /mnt/nas-new/home/zhanggefan/zw/mcl/configs_dota15/mcl/30p.py --launcher pytorch --work-dir result_rph/370 \
--resume-from /mnt/nas-new/home/zhanggefan/zw/mcl/result_rph/370/iter_140800.pth


在/mnt/nas-new/home/zhanggefan/zw/mcl/semi_mmrotate/models/mcl.py中检查数据
水平框的角度为什么不为零

