# 恢复原始数据后的稿件交叉核查补充

## 范围与证据等级

新发现的参考仓库为 `C:\Users\guoqi\OneDrive - stevens.edu\Documents\GitHub\shape-force-est-IMU`，分支 `fix/wrench-conventions`，HEAD=`bc4d13ab8f282ac03ac03a5d248fda995d97c02f`。本次仅读取该副本及其中被忽略的历史输出；未修改、清理、运行其仿真，也未改动稿件。该副本原有三个源码修改（`cosserat_rod_model.py`、`evaluate_kirchhoff_wrench_constrained.py`、`generate_kirchhoff_gt_dataset_constrained.py`）及未跟踪嵌套目录，故 HEAD 或数据meta中的git_hash都不能证明生成时工作树干净。

本补充更新 `manuscript_crosscheck.md` 的检索边界：旧工作区缺少的50例原始GT、序列IMU及稿件结果已在此副本找到。旧报告的“当前HEAD未找到”只适用于旧工作区，不能继续作为整项研究缺少实现或数据的结论。

以下路径均相对于新参考仓库；`CRT`=`src/shape_force_est_imu/crt`，`CMT`=`CRT/gt_data/results/committee_20260426_1647/final_wrench_committee`。稿件仍为临时参考目录 `C:\Users\guoqi\AppData\Local\Temp\oracle_ekf_manuscript_reference_du88vs8z` 中由 `ms.tex:54–56` 激活的三个v2/v3文件。稿件行号均指 `content/sim_study_v3.tex`，除非另行说明。

“确认”指直接读取保存数值/数组/metadata或对保存数组作确定性算术；“来源推定”指日志、源代码与文件组织一致，但没有带源码哈希的逐次运行manifest。本次没有重新运行完整实验，数值匹配不自动验证指标标签正确。

## 1. 稿件主要结果的精确来源

| 稿件结果 | 找到的匹配文件 | 核查结果 |
|---|---|---|
| `tab:wrench_oracle_vs_ekf`，:470–481，12行 | `CMT/oracle_vs_ekf_table.csv:2–13` | 12行六个连续指标均与稿件显示精度一致。对应MD第3行明确N=20 cases×3 noise realizations。 |
| `tab:wrench_oracle_orders`，:443–447，5行 | `CMT/oracle_order_summary.csv:2–6` | 全部匹配；实际n_valid=10。对应MD第3行明确使用`tdcr_gt_samples_10.npz`，不是50例Kirchhoff库。 |
| `tab:wrench_constrained_force_only`，:513–523，10行 | `CMT/constrained_force_only_3d_known_direction_1d_oracle_direction.csv:2–11` | N、方向及归一化列全部匹配；oracle每布局50例，EKF每布局150例。 |
| 跨模型几何 .460/.349 mm，2.483/2.313°，:292 | `CRT/gt_data_seq/results_seq/kirchhoff_shape_est_summary_by_layout_seq.csv:2–3` | 匹配；每布局250运行。实际均值见下。 |
| 可观测性奇异值/条件数，:382–386 | `CRT/gt_data/results/committee_20260426_1737/observability_full/kirchhoff_wrench_observability_summary.csv:2–6` | 50例的中位最小非零奇异值与中位条件数全部匹配。 |
| 分量observable fractions，:413–417 | 同上目录的`wrench_component_observability_summary.csv:2–6` | 各分量均值及weakest标签全部匹配；5D的Fy/Fz来自不同字段，见后文。 |
| 身体坐标约束表，:570–580，10行 | `CRT/gt_data_constrained/results/final_constrained_subspace_20260427/combined_constrained_subspace_summary.csv:2–11` | 绝对误差均值/标准差、百分比及方向角均匹配。oracle50例，EKF250例/布局。 |
| 身体坐标载荷精度，:553–557 | `CRT/gt_data_constrained/kirchhoff_gt_local_transverse_force.npz`及`kirchhoff_gt_local_moment_only.npz` | 直接从保存数组重新计算也匹配，见第4节。 |

关键两行的完整保存值：

| (2,2,1) | tip_mm | 表中rms_mm | dir_F | 表中NRMSE_F | dir_M | 表中NRMSE_M |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | .135959093896885 | **.06726711650585192** | .5299539460927635 | .01153749914389534 | .2948212588221432 | .00521614164675129 |
| EKF 3-IMU | 1.0486267516043826 | .41448609071486736 | **44.24148754643709** | 1.3695467006228352 | 9.162350643211697 | .19319147101921258 |

因此稿件的 **0.07** 和 **44.2°** 不是无法追溯的孤立数字；它们分别是上述列值的舍入。注意下文说明：0.07所在列实际不是数学意义上的RMS。

同名旧文件 `CRT/gt_data/results/final_wrench/oracle_vs_ekf_table.csv` **不匹配**：同阶oracle的rms_mm=.023059341993554、EKF3力方向=61.862900158836°。该目录不能作为当前稿件表的数据源。`CRT/gt_data/results/oracle_vs_ekf_summary.csv` 也是另一轮实验，不能根据相近文件名替代committee表。

## 2. 样本、噪声与运行设置

| 研究 | 保存证据确认的实际数据/设置 | 来源边界 |
|---|---|---|
| 独立oracle阶数扫描 | `artifacts/data/tdcr_gt_samples_10.npz`：T=(10,40,4,4)，seed22、L=.1、meta git_hash=46eb152；阶数(1,1,0)/(1,2,0)/(2,2,0)/(2,2,1)/(3,3,2) | CSV与MD确认10例；命令`--max-cases-oracle 50`只是上限，没有创造50例。其它材料参数由生成表脚本硬编码，不应倒推为旧GT完整生成元数据。 |
| oracle-vs-EKF主表 | `kirchhoff_gt_dataset.npz`前20例，3次静态噪声；布局使用实际采样位置 .5050505/1 与 .2525253/.5050505/1 | committee MD确认20×3；日志及脚本一致支持前20例。不是所有50例，也不是逐步刷新噪声序列。 |
| 几何跨模型研究 | 50例×5序列×30步，γ10、α1、0.5°；P0=[1,.5,1,.5,.25]，Q=[1e-10,1e-12,1e-10,1e-12,1e-10] | `gt_data_seq/results_seq/kirchhoff_shape_est_results_seq.json`的config直接保存。 |
| force-only稿件表 | 50例，3噪声/例，(1,1,0)，静态帧重复更新；oracle50、EKF150/布局 | committee CSV与日志确认数量；30步、γ10、α1、.5°、合成噪声seed99由生成表代码推定。不要套用旁边`constrained_force_only_full`的20步/5噪声/seed999配置。 |
| 身体坐标约束研究 | local transverse与local moment分别50例，GT seed20260427；EKF20步、5噪声、seed999、γ10、α1、.5°、(1,1,0) | 对应`local_*_20260427_20steps_5noise/constrained_results.json` config直接保存；250运行/布局。独立GT结果目录记录steps1、n_noise0。 |

原始通用Kirchhoff GT 的 `positions=(50,100,3)`、`orientations=(50,100,9)`、`tau=(50,4)`，meta记录L=.1 m、半径.0005 m、E=60e9、腱偏置.008 m、40 disks、inextensible=true、tau_max4、力范围±.4 N、矩范围±.04 N·m、min_deflection=.001 m、seed123、git_hash=d58d4f7。这些直接支持稿件:202–240的大部分配置；meta未保存max attempts，不能仅据该文件确认500次上限。

静态IMU文件 `CRT/gt_data/kirchhoff_imu_{2,3}imu.npz` 的 `R_meas=(50,5,n_imu,3,3)`，seed1234。序列文件 `CRT/gt_data_seq/kirchhoff_imu_seq_{2,3}imu.npz` 的 `R_meas_seq=(50,5,30,n_imu,3,3)`，seed_base1234，逐工况/序列递增seed；两者metadata均明确0.5°、左乘三维高斯旋转向量、最近邻映射。主扳手表脚本读前者；跨模型几何论文结果用后者。

跨模型几何的精确均值：2-IMU位置 .45952841899783074 mm、姿态2.483134907991586°；3-IMU位置 .34879243657941217 mm、姿态2.312940322710052°。旧的静态帧`gt_data/results/kirchhoff_shape_est_summary_by_layout.csv`给出 .590729/.485183 mm，不匹配稿件，不能混用。

日志 `CRT/gt_data/results/committee_20260426_1640/run_log.txt:245–250` 记录：`generate_final_wrench_artifacts.py --out-dir "$OUT/final_wrench_committee" --n-cases-ekf 20 --n-noise 3 --max-cases-oracle 50 --max-cases-constrained 50`。日志路径后缀1640与结果目录1647不完全相同，故将其视为同一工作时段的支持证据，而非独立、加密绑定的运行manifest。结果MD和CSV直接确认实际数量。

## 3. 数值找到以后仍然成立的指标问题

1. **RMS标签错误。** `scripts/force_est/generate_final_wrench_artifacts.py:148–160` 的 `shape_metrics` 仅抽取20个弧长点，返回`mean_centerline_error*1000`；`CRT/evaluate_kirchhoff_shape_estimation.py:312`明确是位置误差范数的算术平均。因此表中 .067267→.07 是“平均中心线位置误差”，不是 RMS。该表也未使用几何跨模型节中的全部100点来算此项。
2. **NRMSE标签/归一化与活动稿件不一致。** `CRT/compare_oracle_vs_ekf_wrench_estimation.py:672–688` 返回每例`||估计−GT||/(||GT||+1e-12)`；生成表代码在`:256,351,385,577`汇总这些数的算术均值。它不是稿件:332–346定义的EI/L²、EI/L归一化后整体均方根。找到完全一致的CSV验证了数值来源，不能据此声称稿件NRMSE定义得到验证。
3. **主表静态帧被重复当作独立观测。** 生成表脚本`:315–320,368–376`加载静态R_meas并传入`R_frame`；`compare_oracle_vs_ekf_wrench_estimation.py:645–655`对同一帧重复30次更新。几何研究才使用每步新噪声。稿件:318沿用此前噪声假设的描述不足以区分这两种实验；需要在主表设置中明确说明。
4. **高阶协方差来源现在可定位。** `compare_oracle_vs_ekf_wrench_estimation.py:512–522`直接扩展旧列表：P0额外维度=.25，Q额外维度=1e-12；初值全零（`:645`）。这不是按每个新系数的轴和阶次重建协方差，且稿件未说明；是否适当由诊断报告另评估。
5. **两个oracle表的差异已得到数据来源解释。** 阶数扫描是旧TDCR10，而oracle-vs-EKF是新Kirchhoff前20例；不能继续把两表相同阶数的不同数值当作算术矛盾。活动稿件应明确写出两套数据和样本数。
6. **Fy/Fz标签不是简单转录错误。** component CSV在5D行同时记录`weakest_dominant_component=Fy`、`null_dominant_component=Fz`；前者是最弱非零奇异方向，后者是零空间主分量。稿件:413抄入前者而正文讨论后者，需要解释字段区别。观测子空间使用moment_scale1、force_scale10、numerical rank相对阈值1e-6、effective rank阈值.01；这些来自observability JSON config。

以上源代码证据来自新副本当前文件；对于未修改的生成表入口可直接固定其SHA256。对原有修改的约束评估器及身体坐标生成器，不把当前字节内容未经证明地归到干净bc4d13a提交。父任务的历史审计及小规模复核提供更强的运行验证；此处不把读取文件冒充原始实验重跑。

## 4. 身体坐标数据的直接核验

对保存的local transverse数据计算 `f_body=R_tip.T@f_world`：mean(|Fz|/||F||)=**2.2079520784168795e-11**，max=**1.0789790700966684e-10**，与稿件:553–555的2.21e-11/1.08e-10一致。对local moment数据，保存世界力严格为零；`max ||R_tip.T@moment_world−moment_body||`=**1.3307715404257154e-11 N·m**，与:557一致。

这两套NPZ均为50例×100点，额外保存`f_ext_body/l_ext_body`，meta标记tip_body、seed20260427、git_hash=bc4d13a、min_deflection=.0005。力与矩的世界分量仍单独保存，因此不能只读总括`wrench_frame=world`而忽略更具体的force_frame/moment_frame字段。

## 5. 文件指纹（SHA256）

以下指纹固定本次实际读取的字节；`CMT`和`CRT`含义见开头。

| 文件 | SHA256 |
|---|---|
| `CMT/oracle_vs_ekf_table.csv` | `2f15da9e7e7b4d659d306718b636cfb563cbb52d125a6bbbd832872932c421a2` |
| `CMT/oracle_order_summary.csv` | `451986a236f233f2557e9d3b82e6eb6a151257aa0b99a86b5f2777892f5e9e31` |
| `CMT/constrained_force_only_3d_known_direction_1d_oracle_direction.csv` | `d4bef40cf7e9acb423f3c01cdb5e359e5b0c558c6dfc66a42a5dac2a2a50b113` |
| `CRT/gt_data/results/final_wrench/oracle_vs_ekf_table.csv`（不匹配） | `6a6cdb14cd15e6dbf1214064e5b86f19b5f31e071ac81ce5183e1a97b1523e64` |
| `CRT/gt_data_seq/results_seq/kirchhoff_shape_est_summary_by_layout_seq.csv` | `e76a5e629d8c2fbe3704be4ae2436847b0b5987146fda5abc728385f083730e7` |
| `CRT/gt_data_constrained/results/final_constrained_subspace_20260427/combined_constrained_subspace_summary.csv` | `da8332650475bf649746b0ebf1c6e9ab0db14631e77eb33298a4b85d84d507b0` |
| `CRT/gt_data/results/committee_20260426_1737/observability_full/kirchhoff_wrench_observability_summary.csv` | `077152c548831c75df8e448da79e298eeeb34d0dfd4ab036a862b3b5d20ea56f` |
| 同目录`wrench_component_observability_summary.csv` | `5d7806996312aff7ee1e403ce255f1971ba43e2ad2215949e091c1c50fb03b13` |
| `CRT/gt_data/kirchhoff_gt_dataset.npz` | `ae8ec9a3629a136e02b7919bb47eabe537e80e448457295fe9465494a608ba3a` |
| `CRT/gt_data/kirchhoff_imu_2imu.npz` | `c0f47dc6d509d6d68d47796ccc6a2acb8c99762e0cd26d75ba1815f157c16e4b` |
| `CRT/gt_data/kirchhoff_imu_3imu.npz` | `ef709d85eb766e58dd8cf1358fd089a1623931c576ed857ecfcfdd7f026fd6e3` |
| `CRT/gt_data_seq/kirchhoff_imu_seq_2imu.npz` | `bc98daea1ed9cd3c4ea050191b8de5e4a3a393c0475eecd6295c57b63b880d67` |
| `CRT/gt_data_seq/kirchhoff_imu_seq_3imu.npz` | `5772c840b83a99e08c3ffe82d89462567c04c9a18609acd693e8c44d04398338` |
| `artifacts/data/tdcr_gt_samples_10.npz` | `aaaa5b1ad51336503a1ca53f9397bfa4c8380c23637ef41b0b4c2e33ae1524a7` |
| `CRT/gt_data_constrained/kirchhoff_gt_force_only.npz` | `74c7d453c2329fca09e9b17d8f707e836662af10eaa3a7c87124ccd21b510c8b` |
| `CRT/gt_data_constrained/kirchhoff_gt_local_transverse_force.npz` | `441986a80b252c3d65eb2a72b9ea88e1668c360fc9910536c7d19f6be4f2893e` |
| `CRT/gt_data_constrained/kirchhoff_gt_local_moment_only.npz` | `361e4fb4b01385373726d55b203cae581e7578f2c9a45aeda4358460e3788199` |
| `scripts/force_est/generate_final_wrench_artifacts.py` | `eee037f1f260d8b3bf7e09d3dd9410a04ff070e92303f86a0e49d3850d77b3de` |
| `CRT/gt_data/results/committee_20260426_1640/run_log.txt` | `b12d56745c3f18decbbedfaaef739f64d6ad1923b8a70abef9e1bba45f83c98f` |

结论：当前稿件的大部分后期结果现在已有精确保存来源，并且身体坐标数据的关键校验值能够直接重算确认。最需要修正的不是数值“丢失”，而是主表RMS/NRMSE标签、20例×3静态噪声的实验范围、两套oracle数据源，以及静态扳手实验与序列几何实验的区别。
