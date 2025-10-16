cd /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl
conda activate zw_mr

单分支(ratio_range)：
nohup bash -c 'CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25540 \
    train.py configs_dota15/mcl/12510_onebranch_rr.py \
    --launcher pytorch \
    --work-dir zw_result/data2_onebr_rr_2001' \
> ./zz_log/data2_onebr_rr_2001.log 2>&1 &

单分支无prompt(ratio_range)：
nohup bash -c 'CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25520 \
    train.py configs_dota15/mcl/12510_onebranch_wo_p_rr.py \
    --launcher pytorch \
    --work-dir zw_result/data2_onebr_wo_p_rr_2001_1' \
> ./zz_log/data2_onebr_wo_p_rr_2001_1.log 2>&1 &

单分支无prompt(没有ratio_range) 旋转框：
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25500 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/12510_onebranch_wo_p_wo_rr_rbox.py \
    --launcher pytorch \
    --work-dir zw_result/data2_onebr_wo_p_wo_rr_rbox_2050' \
> ./zz_log/data2_onebr_wo_p_wo_rr_rbox_2050.log 2>&1 &

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
单分支无prompt(ratio_range) 旋转框：
nohup bash -c 'CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25510 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/12510_onebranch_wo_p_rr_rhp.py \
    --launcher pytorch \
    --work-dir zw_result/data2_onebr_wo_p_rr_rhp_3020' \
> ./zz_log/data2_onebr_wo_p_rr_rhp_3020.log 2>&1 &

单分支无prompt(ratio_range) 旋转框：dota1ss1
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25530 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/12510_onebranch_wo_p_rr_rbox_dota1ss1.py \
    --launcher pytorch \
    --work-dir zw_result/data2_onebr_wo_p_rr_rbox_ss1_3010' \
> ./zz_log/data2_onebr_wo_p_rr_rbox_ss1_3010.log 2>&1 &


mcl 旋转框对比试验：  1.0 
nohup bash -c 'CUDA_VISIBLE_DEVICES=2 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=1 \
    --master_port=25560 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/10_mcl_fcos_dota.py \
    --launcher pytorch \
    --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_mcl_2010_1' \
> /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zz_log/data2_mcl_2010_1.log 2>&1 &

nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25500 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/10_mcl_fcos_dota.py \
    --launcher pytorch \
    --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_mcl_2010_2ka' \
> /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zz_log/data2_mcl_2010_2ka.log 2>&1 &

mcl 旋转框对比试验：  1.0
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25560 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/10_mcl_fcos_dota.py \
    --launcher pytorch \
    --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_mcl_2050' \
> /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zz_log/data2_mcl_2050.log 2>&1 &

mcl 旋转框对比试验：  1.5
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=1 \
    --master_port=25580 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/15_mcl_fcos_dota.py \
    --launcher pytorch \
    --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_mcl_3020_15' \
> /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zz_log/data2_mcl_3020_15.log 2>&1 &



对比实验：
单分支无prompt(ratio_range) 旋转框 PWOOD选点：
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25520 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/12510_onebranch_wo_p_rr_rbox_pwoodSelect.py \
    --launcher pytorch \
    --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_onebr_wo_p_rr_rbox_pwSe_3020' \
> /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zz_log/data2_onebr_wo_p_rr_rbox_pwSe_3020_1.log 2>&1 &

对比实验：
单分支无prompt(ratio_range) 旋转框 rsst选点：
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25520 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/12510_onebranch_wo_p_rr_rbox_rsstSelect.py \
    --launcher pytorch \
    --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_onebr_wo_p_rr_rbox_rsSe_2010' \
> /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zz_log/data2_onebr_wo_p_rr_rbox_rsSe_2010.log 2>&1 &
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pwood 旋转框：
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25570 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/30p_rbox.py \
    --launcher pytorch \
    --work-dir zw_result/data2_pwood_rbox_3020' \
> ./zz_log/data2_pwood_rbox_3020.log 2>&1 &
pwood 水平框：
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25570 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/30p_hbox.py \
    --launcher pytorch \
    --work-dir zw_result/data2_pwood_hbox_3020' \
> ./zz_log/data2_pwood_hbox_3020.log 2>&1 &
pwood 点：
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25540 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/30p_point.py \
    --launcher pytorch \
    --work-dir zw_result/data2_pwood_point_1010' \
> ./zz_log/data2_pwood_point_1010.log 2>&1 &


RSST 两分支作为baseline: DOTA1.0
cd /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/RSST/RSST
nohup bash -c 'CUDA_VISIBLE_DEVICES=3 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=1 \
    --master_port=25540 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/RSST/RSST/config_sparse/rsst/rsst_fcos_dota_percent1_with_pseudo_labeled_data_2branch.py \
    --launcher pytorch \
    --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_baseline_rsst_2020' \
> /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zz_log/data2_baseline_rsst_2020.log 2>&1 &

nohup bash -c 'CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25520 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/RSST/RSST/config_sparse/rsst/rsst_fcos_dota_percent1_with_pseudo_labeled_data_2branch.py \
    --launcher pytorch \
    --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_baseline_rsst_2050' \
> /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zz_log/data2_baseline_rsst_2050.log 2>&1 &


RSST 两分支作为baseline: DOTA1.5
cd /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/RSST/RSST
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=1 \
    --master_port=25510 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/RSST/RSST/config_sparse/rsst/rsst_fcos_dota_percent1_with_pseudo_labeled_data_2branch_for15.py \
    --launcher pytorch \
    --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_baseline_rsst_2010_15' \
> /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zz_log/data2_baseline_rsst_2010_15.log 2>&1 &

nohup bash -c 'CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25550 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/RSST/RSST/config_sparse/rsst/rsst_fcos_dota_percent1_with_pseudo_labeled_data_2branch_for15.py \
    --launcher pytorch \
    --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_baseline_rsst_3010_15' \
> /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zz_log/data2_baseline_rsst_3010_15.log 2>&1 &


单分支无prompt(ratio_range) + cls_loss：
nohup bash -c 'CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25550 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/12510_onebranch_wo_p_rr_sflp_rbox.py \
    --launcher pytorch \
    --work-dir zw_result/data1_onebr_wo_p_rr_sflp_rbox_3010' \
> ./zz_log/data1_onebr_wo_p_rr_sflp_rbox_3010.log 2>&1 &

单分支无prompt + 自监督loss：
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25550 \
    train.py configs_dota15/mcl/12510_onebranch_wo_p_res_rr.py \
    --launcher pytorch \
    --work-dir zw_result/data2_onebr_wo_p_resall_rr_2050' \
> ./zz_log/data2_onebr_wo_p_resall_rr_2050.log 2>&1 &

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
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25550 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/12510_onebranch_wo_p_rr_rbox_dota1ss1.py \
    --resume-from /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_onebr_wo_p_rr_rbox_ss1_3010/iter_73600.pth \
    --launcher pytorch \
    --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_onebr_wo_p_rr_rbox_ss1_3010' \
> /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zz_log/data2_onebr_wo_p_rr_rbox_ss1_3010_2.log 2>&1 &

resume
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=1 \
    --master_port=25510 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/15_mcl_fcos_dota.py \
    --resume-from /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_mcl_3020_15/iter_99200.pth \
    --launcher pytorch \
    --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_mcl_3020_15' \
> /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zz_log/data2_mcl_3020_15.log 2>&1 &

RSST 旋转框测试
cd /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/RSST/RSST
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=25500 \
    train.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/RSST/RSST/config_sparse/rsst/rsst_fcos_dota_percent1_with_pseudo_labeled_data_3branch.py \
    --launcher pytorch \
    --work-dir /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_rsst_2020_2' \
> /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zz_log/data2_rsst_2020_2.log 2>&1 &




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
cd /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/RSST/RSST
python test.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/RSST/RSST/config_sparse/rsst/rsst_fcos_dota_percent1_with_pseudo_labeled_data_2branch.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result_toAnalyze/data2_baseline_rsst_2050/best_0.595138_mAP.pth --format-only --eval-options submission_dir=/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result_toAnalyze/data2_baseline_rsst_2050/task1_data2_baseline_rsst_2050

cd /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl
python test.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/12510_onebranch_wo_p_rr_rhp.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_onebr_wo_p_rr_rhp_3020/best_0.604927_mAP.pth --format-only --eval-options submission_dir=/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/zw_result/data2_onebr_wo_p_rr_rhp_3020/task1_data2_onebr_wo_p_rr_rhp_3020



有标签测试（train/val）：
python test.py /inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/mcl/configs_dota15/mcl/12510.py /mnt/nas-new/home/zhanggefan/zw/mcl/zz_result1/mclrsst3a/best_0.633173_mAP.pth --eval mAP

在/mnt/nas-new/home/zhanggefan/zw/mcl/semi_mmrotate/models/mcl.py中检查数据
水平框的角度为什么不为零

