cd /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl
conda activate zw_mr

单分支：
nohup bash -c 'CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25500 \
    train.py configs_dota15/mcl/12510_onebranch.py \
    --launcher pytorch \
    --work-dir zw_result/data2_onebr_3010_1' \
> ./zz_log/data2_onebr_3010_1.log 2>&1 &

单分支无prompt：
nohup bash -c 'CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25530 \
    train.py configs_dota15/mcl/12510_onebranch_wo_p.py \
    --launcher pytorch \
    --work-dir zw_result/data2_onebr_wo_p_3010' \
> ./zz_log/data2_onebr_wo_p_3010.log 2>&1 &

单分支无prompt + 自监督loss：
nohup bash -c 'CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25530 \
    train.py configs_dota15/mcl/12510_onebranch_wo_p_res.py \
    --launcher pytorch \
    --work-dir zw_result/data2_onebr_wo_p_resall_2001' \
> ./zz_log/data2_onebr_wo_p_resall_2001.log 2>&1 &

单分支无prompt + 自适应选点：
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25530 \
    train.py configs_dota15/mcl/12510_onebranch_wo_p_adpinds.py \
    --launcher pytorch \
    --work-dir zw_result/data2_onebr_wo_p_adpinds_2001' \
> ./zz_log/data2_onebr_wo_p_adpinds_2001.log 2>&1 &

单分支无prompt + 自适应选点 + 自监督loss：
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25510 \
    train.py configs_dota15/mcl/12510_onebranch_wo_p_adpinds_res.py \
    --launcher pytorch \
    --work-dir zw_result/data2_onebr_wo_p_adpinds_res_2001' \
> ./zz_log/data2_onebr_wo_p_adpinds_res_2001.log 2>&1 &

监督sparse双分支： 
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=4 \
    --master_port=25530 \
    train.py configs_dota15/mcl/12510_supsparse.py \
    --launcher pytorch \
    --work-dir zw_result/data2_sup_2050' \
> ./zz_log/data2_sup_2050.log 2>&1 &

传统双分支： 
nohup bash -c 'CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25540 \
    train.py configs_dota15/mcl/12510.py \
    --launcher pytorch \
    --work-dir zw_result/data2_sup_3010' \
> ./zz_log/data2_sup_3010.log 2>&1 &

resume
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=4 \
    --master_port=25520 \
    train.py configs_dota15/mcl/12510.py \
    --resume-from /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_ori_2050_4_all_6400/iter_89600.pth\
    --launcher pytorch \
    --work-dir zw_result/data2_ori_2050_4_all_6400' \
> ./zz_log/data2_ori_2050_4_all_6400.log 2>&1 &




cd /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/RSST/RSST

nohup bash -c 'CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25500 \
    train.py config_sparse/rsst/rsst_fcos_dota_percent1_with_pseudo_labeled_data_3branch.py \
    --launcher pytorch \
    --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_rs_3010_12800' \
> /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zz_log/data2_rs_3010_12800.log 2>&1 &



cd /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/sood-mcl

nohup bash -c 'CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25510 \
    train.py configs_dota15/mcl/mcl_fcos_dota15_20p.py \
    --launcher pytorch \
    --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_mcl_3010' \
> /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zz_log/data2_mcl_3010.log 2>&1 &


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --master_port=25500 \
train.py configs_dota15/mcl/12510.py \
 --launcher pytorch --work-dir zw_result/unsuploss_12800

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --master_port=25550 \
train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/30p.py \
--launcher pytorch --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/zzzz \
--resume-from /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_p_s/iter_28800.pth

无标签测试（test）：注意the out_folder should be a non-exist path
python test.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/12510_onebranch_wo_p_adpinds_res.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_onebr_wo_p_adpinds_res_2050/best_0.615146_mAP.pth --format-only --eval-options submission_dir=/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_onebr_wo_p_adpinds_res_2050/task1_data2_onebr_wo_p_adpinds_res_2050

有标签测试（train/val）：
python test.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/12510.py /mnt/nas-new/home/zhanggefan/zw/mcl/zz_result1/mclrsst3a/best_0.633173_mAP.pth --eval mAP


在/mnt/nas-new/home/zhanggefan/zw/mcl/semi_mmrotate/models/mcl.py中检查数据
水平框的角度为什么不为零

