cd /mnt/nas-new/home/zhanggefan/zw/mcl
conda activate zw_mcl

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --master_port=25510 \
train.py /mnt/nas-new/home/zhanggefan/zw/mcl/configs_dota15/mcl/30p.py --launcher pytorch --work-dir zz_result/mcl/


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --master_port=25550 \
train.py /mnt/nas-new/home/zhanggefan/zw/mcl/configs_dota15/mcl/30p.py --launcher pytorch --work-dir result_rph/370 \
--resume-from /mnt/nas-new/home/zhanggefan/zw/mcl/result_rph/370/iter_140800.pth


无标签测试（test）：注意the out_folder should be a non-exist path
python test.py /mnt/nas-new/home/zhanggefan/zw/mcl/configs_dota15/mcl/12510.py /mnt/nas-new/home/zhanggefan/zw/mcl/zz_result1/mclrsst3a/best_0.633173_mAP.pth --format-only --eval-options submission_dir=/mnt/nas-new/home/zhanggefan/zw/mcl/zz_result1/mclrsst3a/task1
有标签测试（train/val）：
python test.py /mnt/nas-new/home/zhanggefan/zw/mcl/configs_dota15/mcl/12510.py /mnt/nas-new/home/zhanggefan/zw/mcl/zz_result1/mclrsst3a/best_0.633173_mAP.pth --eval mAP


在/mnt/nas-new/home/zhanggefan/zw/mcl/semi_mmrotate/models/mcl.py中检查数据
水平框的角度为什么不为零

