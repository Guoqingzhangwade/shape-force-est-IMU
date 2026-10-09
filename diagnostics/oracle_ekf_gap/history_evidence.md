# 历史实现与论文结果溯源证据

本文的“当前 HEAD”指未包含本次工作树修复的 `d58d4f7`；两处后续最小修复及数值验证见 [diagnosis.md](diagnosis.md)。

审计日期：2026-10-08。本文仅记录读取本地 Git 对象所得证据；没有切换、合并、拉取分支，也没有执行历史仿真。本文提到的历史文件并不存在于当前工作树时，行号均指下述固定历史版本，可用 `git show <完整提交>:<路径>` 核对。

## 版本与结论边界

| 角色 | 已核实版本 |
|---|---|
| 当前工作树 HEAD | `d58d4f795adce27fdbc7a54dd99efb81bbe3df8f` |
| 已存在的远端跟踪引用 `origin/fix/wrench-conventions` | `bc4d13ab8f282ac03ac03a5d248fda995d97c02f` |
| 原始跨模型流程提交 | `78454849b5e5a18bfeb796d47834d75b16b4e848` |
| Oracle/EKF 与汇总表脚本首次出现 | `c0d8687c1c1e535cb6e361efd6f3a8a0cb6180f5` |
| 汇总表脚本最后修改 | `67b28ecc2b08bfbeefe35a9ac3bc0e19d41dc68f` |
| Oracle/EKF 脚本最后修改 | `ed41bbc31a2efad9b4fdebfc987b8d9b5b130864` |

**结论：8D、三 IMU 的原始候选实现存在于本地可读的历史分支；当前分支缺少这套实现。** 不能写成“整个仓库历史从未实现 8D EKF”。历史汇总脚本的表格结构与论文高度吻合，但没有原始输入、运行命令及生成表格，尚不能确认论文数值由哪个提交、哪些参数产生。

以下用短名表示固定版本 `bc4d13ab8f282ac03ac03a5d248fda995d97c02f` 的文件：

- `FINAL`：`scripts/force_est/generate_final_wrench_artifacts.py`
- `COMPARE`：`src/shape_force_est_imu/crt/compare_oracle_vs_ekf_wrench_estimation.py`
- `SHAPE`：`src/shape_force_est_imu/crt/evaluate_kirchhoff_shape_estimation.py`
- `SEQUENCE`：`src/shape_force_est_imu/crt/evaluate_kirchhoff_shape_estimation_sequences.py`
- `IMU`：`src/shape_force_est_imu/crt/generate_kirchhoff_imu_measurements.py`
- `VW`：`src/shape_force_est_imu/crt/virtual_work.py`

## 候选汇总入口与实验设置

`FINAL:302–428` 的 `run_oracle_vs_ekf` 生成 `oracle_vs_ekf_table.csv/.md`。其列为 order、n、method、tip_mm、rms_mm、dir_F、NRMSE_F、dir_M、NRMSE_M（399–425），与活动论文 `content/sim_study_v3.tex:461–481` 的 Oracle/EKF 表一致。

| 项目 | 历史代码证据 | 审计含义 |
|---|---|---|
| 阶数 | `FINAL:83–88` 包括 `(1,1,0)`、`(2,2,0)`、`(2,2,1)`、`(3,3,2)` | 8D 和 11D 路径已实现；不能仅凭 `COMPARE` CLI 最大 8D 就断言 11D 不存在 |
| 默认样本数 | `FINAL:90–91`：`N_CASES_EKF=3`、`N_NOISE=5` | 默认仅 3 个形状；不应把论文别处的 50 个 GT 样本直接当作该表使用的样本数 |
| 实际截取 | `FINAL:315–326` 取请求数量与两个 IMU 文件可用数量的最小值，随后取前若干样本 | 缺原始命令与文件，不能断言实际执行了默认 3×5 |
| 布局 | `FINAL:77–78`：2 IMU `{0.50,1.00}`；3 IMU `{0.25,0.50,1.00}` | 317–318 优先采用 NPZ 元数据 `imu_actual_s`；标称与离散后的实际位置可能不同 |
| 更新数与噪声 | `FINAL:371–376` 使用 `SHAPE:74–84` 的常量：α=1、0.5°、30 步、γ=10 | 不能沿用论文 matched-model 的 50 步来解释此入口 |
| 先验 | `COMPARE:512–522` 把 5D 的 P0/Q 列表直接向后补齐 | 是具体实现选择，并非论文提供了完整的 8D 协方差定义 |
| 异常处理 | `FINAL:354–355,388–389` 为 `except Exception: pass` | 失败样本可被静默丢弃；该表不输出成功计数 |
| 分开的 Oracle 表 | `FINAL:48–52,208–218,278`：独立阶数 Oracle 表读取 `tdcr_gt_samples_10.npz`；Oracle/EKF 表读取 Kirchhoff GT | 两张 Oracle 表使用不同数据，不宜直接比较数值作为同一次试验 |

覆盖设置已检查：`FINAL:759–785` 仅提供 `--out-dir`、`--dry-run`、`--n-cases-ekf`、`--n-noise`、两个样本上限参数。`FINAL:94–99` 的 `expandvars/expanduser` 只展开输出路径。该文件没有读取 `os.environ` 或 `os.getenv` 来覆盖更新数、噪声、数据路径或物理常量。输入路径固定在 48–52 行，α、更新数等由导入常量决定；外部包装程序或当时未提交修改仍不能排除。

## 指标定义不一致

1. **RMS 列实际为均值。** `FINAL:148–160` 只选 20 个近似均匀的离散位置，返回 `g["mean_centerline_error"]`，随后以 `rms_mm` 写出。`SHAPE:312` 明确计算 `float(np.mean(pos_err))`。这不是 `sqrt(mean(pos_err**2))`。论文 `sim_study_v3.tex:461,468` 把该列称为 RMS。

2. **NRMSE 列实际为平均相对范数误差。** `COMPARE:678–690` 的 `nrmse_force = force_err/(||f_gt||+1e-12)`，moment 同理；`FINAL:358–359,390–393` 再取算术平均。论文 `sim_study_v3.tex:339–346` 定义固定特征尺度 F0=EI/L²、M0=EI/L 下的归一化误差及聚合 NRMSE。候选代码既不是该固定尺度，也未执行平方平均再开方。

3. **零向量方向处理不同。** `COMPARE:665–668` 在估计或真值范数小于 1e-12 时返回 0°；论文 `sim_study_v3.tex:360` 表示接近零的真值应从方向统计中排除。零返回值可能人为降低聚合方向误差。

这些是候选历史实现中可直接证实的问题。因为生成表格未保存，不能进一步断言论文每一个数值必然由上述错误定义计算。

## 相同观测重复更新与逐步新噪声不是同一实验

`FINAL:368–369` 为每个 case/noise 取一个静态 `R_frame`，交给 `COMPARE.run_ekf_for_order`。`COMPARE:648–656` 循环所有更新时始终使用同一个 `R_frame`，没有逐步获取新的 IMU 样本。

`IMU:161–171` 每个 realization/sensor 仅抽一次噪声旋转，并广播到所有 GT cases。该入口既没有独立的每步噪声，也没有独立的每个 case 噪声抽样。

历史中另有独立的 `SEQUENCE` 路径：其 123–124 行按 `step_idx` 取 `R_seq[step_idx,sensor_idx]`，每步读取不同保存帧。该文件沿用 `SHAPE` 的 5D `STATE_DIM`（120 行初始化）。不能据此声称候选 8D wrench 表已经改用逐步独立噪声序列。GT 仍为静态形状；改变的是测量处理方式，不是动态杆运动。

## Oracle 拟合目标也必须区分

`COMPARE:384–428` 的 dense Oracle 先用相邻姿态 `log(R_i.T R_(i+1))/ds` 求曲率，再对各轴的多项式系数执行 `np.linalg.lstsq`。位置数组只用来取得采样点数量，并不进入位置残差。`FINAL:344–345` 明确走这一路径。

当前 HEAD 的 `scripts/force_est/force_est_curve_fit.py:45–52` 默认目标为加权位置平方误差加姿态角平方误差；164–167 行默认两权重均为 1；281–287 行用 L-BFGS-B。当前代码也存在 `--use-gt-curvature`（174–175、275–278 行），因此准确说法是“默认 pose-fit 与历史表格的 curvature-fit 不同”，而不是“当前没有 curvature-fit”。本次单样本 pose-fit 诊断不能冒充历史表格的 dense Oracle 复现。

## 后续 tendon 修正使历史提交不可混用

`c0d8687` 中已有上述候选表格入口，之后 mechanics 又发生实质变化：

- `4362865b811076a9e8ca78c895f7b02ff163a0fd` 修改 `virtual_work.py`，将归一化弧长下 tendon 切向量从 `e3 + kappa×r` 改为 `L*e3 + kappa×r`，并明确 `J_qm = -dℓ/dm`。固定历史 tip 的 `VW:186–197,235–243` 可见新定义。
- `26382ce9833dd73daeaf032fed9113423d75f7cd` 继续修改相关 wrench study/subspace helper。
- `d8792d920c9e67a4a4ee3dad7ec12502d9bfb5d3` 将活动 wrench 脚本调用从 `cable_jacobian` 改为 `pull_jacobian`；`COMPARE.wrench_from_modal` 随之变更。

这里不宜只说“旧版符号反了”：旧 `cable_jacobian` 名称虽然暗示 cable length，其已有积分负号；修正还包含明确的物理长度尺度。需要按同一提交成套读取调用者和 `virtual_work.py`。

当前 HEAD 的 `virtual_work.py:177–178` 仍为 `e3 + hat(kappa) @ r_i`。因此当前小诊断、`c0d8687` 候选脚本和 `bc4d13a` 修正后脚本并非完全相同的 mechanics。即使后来修正后的脚本能执行，也不能据此认定它就是论文数字的生成版本。

## 缺失输入与结果的检查范围

执行了历史 tip 的递归 `git ls-tree`，以及 `git log --all` 对下列目录/精确文件名的历史查询，未发现它们曾在本地可见引用中被跟踪：

- `src/shape_force_est_imu/crt/gt_data/kirchhoff_gt_dataset.npz`
- 同目录 `kirchhoff_imu_2imu.npz`、`kirchhoff_imu_3imu.npz`
- `gt_data/results/final_wrench/oracle_vs_ekf_table.csv/.md`、`oracle_order_summary.csv`
- `oracle_vs_ekf_results.json`；`gt_data`、`gt_data_seq`、`gt_data_constrained` 下的版本历史

历史 tip 的 `.gitignore` 明确忽略这三个 GT 目录。`artifacts/data/tdcr_gt_samples_10.npz` 则确实已跟踪，但它不是 Oracle/EKF 表需要的 Kirchhoff GT/IMU 数据组合。

“缺失”仅指当前工作树检查及本机现有 Git 引用/历史中的上述路径与文件名；本次未 fetch，也没有证明远端未获取分支、其他设备、其他名称的归档或私人目录里不存在这些文件。

要核实论文 8D/三 IMU 表行，至少需要恢复原始 Kirchhoff GT、3-IMU NPZ、其元数据、精确代码提交、运行参数及对应 CSV/JSON。要完整运行 `FINAL`，还需要 2-IMU 文件及 constrained GT（48–52、802–806 行）。原始行级结果优于只有汇总均值的表格。

## 有界处理决定

可以把固定历史提交的 Python 文件物化到仓库外临时目录，从技术上运行 8D 实现；但没有原始 GT/IMU 数据，生成替代数据只会形成新实验。因此本次不运行该历史完整流程，不替用户生成“论文复现结果”，也不把临时稿件加入仓库。

本次已有的单样本当前代码诊断只用于定位 mechanics/拟合/维数兼容性问题。论文所列 `(2,2,1)` 三 IMU 的 44.2°、NRMSE-F 1.370 等数值（`sim_study_v3.tex:478`）仍应标记为“找到了候选实现，但输入与数值溯源未恢复”。
