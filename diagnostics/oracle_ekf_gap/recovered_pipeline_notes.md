# 找回副本后的原始 8D 流程补充审计

日期：2026-10-08。只读来源为 `C:\Users\guoqi\OneDrive - stevens.edu\Documents\GitHub\shape-force-est-IMU`（下称 NEW）；本文件写在原工作区。没有运行估计实验、生成数据、修改 NEW 或切换其分支。

**此前“原始数据未找到”的阻塞已解除。** `history_evidence.md` 中关于本地 Git 跟踪历史的结论仍成立，但不能再把输入/结果文件描述为所有已知副本中都缺失。NEW 中找到了被 Git 忽略的 GT、静态 IMU、逐步 IMU 序列和多组历史结果。

## 代码边界

NEW 分支为 `fix/wrench-conventions`，HEAD=`bc4d13ab8f282ac03ac03a5d248fda995d97c02f`。现有修改为 `cosserat_rod_model.py`、`evaluate_kirchhoff_wrench_constrained.py`、`generate_kirchhoff_gt_dataset_constrained.py`，另有未跟踪的 `src/shape_force_est_imu/crt/src/`。本次均保持原样。

以下四个直接相关文件已用 `git diff --quiet HEAD -- <file>` 核实工作树未改动：

- `src/shape_force_est_imu/crt/compare_oracle_vs_ekf_wrench_estimation.py`（下称 COMPARE）
- 同目录 `evaluate_kirchhoff_shape_estimation.py`（SHAPE）
- 同目录 `virtual_work.py`（VW）
- 同目录 `sweep_ekf_wrench_convergence.py`（SWEEP）

直接从保存 GT 调用这些 API 不依赖三个已修改的杆生成/约束评估文件，无需重跑求解器。不得把 NEW 的 dirty 状态概括为“整个流程都是干净提交”；运行清单应保存分支、HEAD、完整 status 和实际导入文件哈希。

## 输入 schema、实际位置与随机性

下表路径均相对 NEW 的 `src/shape_force_est_imu/crt/`。

| 文件 | 已读取的数组与元数据 |
|---|---|
| `gt_data/kirchhoff_gt_dataset.npz` | 50 cases；`case_id:(50,)`、`tau:(50,4)`、`f_ext/l_ext:(50,3)`、`positions:(50,100,3)`、`orientations:(50,100,9)`、`T_tip:(50,4,4)` |
| `gt_data/kirchhoff_imu_3imu.npz` | `R_true:(50,3,3,3)`；`R_meas:(50,5,3,3,3)`，轴为 case/noise/sensor/rotation；5 个静态噪声 realization |
| `gt_data_seq/kirchhoff_imu_seq_3imu.npz` | `R_true:(50,3,3,3)`；`R_meas_seq:(50,5,30,3,3,3)`，增加 step 轴；`noise_seed_bank:(50,5)` |

三者的 `case_id` 完全相同，从 0 开始；两个 IMU 文件的 `R_true` 逐元素相同。标称位置为 `[0.25,0.5,1]`，采样索引 `[25,50,99]`；应传给原估计器的实际位置是 `[25/99,50/99,1]`，即 `[0.25252525252525254,0.5050505050505051,1]`。

GT 元数据：L=0.1 m、E=60 GPa、backbone radius=0.0005 m、tendon offset=0.008 m、40 disks、seed=123；生成日期 `2026-04-19T18:51:44`，记录的 `git_hash="d58d4f7"`。该 git_hash 只能作为数据内保存的来源线索，不能证明当时无未提交改动。

静态 IMU 元数据：0.5°、seed=1234、生成日期 `2026-04-19T20:39:10`。其生成代码按 realization/sensor 抽取噪声并广播到所有 cases；没有逐 case 的独立 seed bank。直接读取保存的 `[case=0,noise=0]` 最能保持原始输入，不能仅靠重新设 seed=1234 就假设可重建同一个三 IMU 帧（生成器先前布局消耗的 RNG 也会影响抽样）。

序列元数据：0.5°、30 步、seed_base=1234，`seed_bank[case,noise]=1234+case*5+noise`；前两行实际为 `[1234..1238]` 和 `[1239..1243]`。序列日期 `2026-04-20T18:22:35`。它保持 GT 静态，每步抽取新噪声。静态 `[0,0]` 与序列 `[0,0,0]` 并不相等，最大旋转矩阵元素差约 0.0327223；因此切换数据文件也同时换了噪声实现。

## 最小原始 8D／三 IMU 调用

固定 `case_index=0`、`case_id=0`、`noise_index=0`、`order=(2,2,1)`；不重新采样，也不运行脚本 main。

```python
R_frame = list(imu["R_meas"][0, 0])
s_imu = np.asarray(imu_meta["imu_actual_s"])
m_ekf = COMPARE.run_ekf_for_order(
    R_frame=R_frame, imu_pos_norm=s_imu,
    order_x=2, order_y=2, order_z=1,
    gamma=10, meas_std_deg=0.5, alpha=1.0, steps=30,
)
m_oracle = COMPARE.estimate_modal_oracle(
    gt["positions"][0], gt["orientations"][0], 2, 2, 1, L_phys=0.1,
)
f_est, mu_est = COMPARE.wrench_from_modal(m_ekf, gt["tau"][0], 2, 2, 1, gamma=10)
```

`run_ekf_for_order` 签名见 COMPARE:492–502；返回值只有 `m_est`，没有 P 或迭代轨迹。其 648–656 行重复同一静态帧 30 次。初值为 8 个零；P0 对角线为 `[1,.5,1,.5,.25,.25,.25,.25]`；Q 对角线为 `[1e-10,1e-12,1e-10,1e-12,1e-10,1e-12,1e-12,1e-12]`（512–522）。不要为“更自然的轴顺序”暗中改这些默认值。

`estimate_modal_oracle`（384–428）以相邻姿态差分曲率做分轴多项式最小二乘。它不最小化位置残差，且不等同于当前旧工作区默认 pose-fit。`wrench_from_modal` 使用固定 mechanics 常量 L=.1、E=60 GPa、r=.0005、tendon=.008，本数据与之匹配。

完整形状可用现有 `fwd_transform_general(m,s,gamma,L,2,2,1)` 重建。应同时明确记录真正 RMS 与历史表中的平均位置误差；不能沿用 `rms_mm` 名称而不说明历史定义。

## 现有其他 API 的适用边界

`fit_sparse_imu_batch(R_meas, imu_positions, order_x, order_y, order_z, L, gamma, m0=None, sigma_rot_rad=None, prior_mean=None, prior_cov=None, max_iter=20, multistart=1, seed=0)`（COMPARE:276–291）可复用同一静态三 IMU 帧作 sparse batch 基线。残差为 `Log(R_meas @ R_pred.T)/sigma`；若无 prior，不增加先验残差。

**初值陷阱：** COMPARE:319–328 始终把零向量放在第一个 start；只有 `multistart>1` 才把 `m0` 放入第二个 start。因此 `m0=m_ekf, multistart=1` 实际仍从零开始。不得将此结果标为 EKF warm-start。

返回 9 元组（362–371）：`m,cost,success,nfev,residual_norm,solver_name,status,message_short,hit_limit`。`max_iter` 实际传给 SciPy 的 `max_nfev`，不是迭代数；数值 Jacobian 的全部函数调用开销也不等同于 nfev。应记录全部状态；达到上限的 sparse batch 不能作为已收敛 Oracle。

`SWEEP.run_tuned_ekf(R_frame, imu_pos_norm, order, gamma, L_phys, meas_std_deg, alpha, p0_scale, q_scale, steps, init_m=None)` 支持调整先验和初始值，返回 `m,P,final_residual_norm,final_nis`。但 SWEEP:176–186 在非 5D 时使用有限差分 H；原 COMPARE:629–640 使用其自定义解析近似 H。把 SWEEP 作为原始 baseline 会同时更换 Jacobian，必须作为单独的诊断变体标记。

现有 sequence evaluator 沿用 5D SHAPE 状态；它不是直接可调用的原 8D sequence baseline。不能把静态原 8D 流程、SWEEP 有限差分变体、5D sequence evaluator 当作同一实现。

## 找回的保存结果并未恢复论文数值一致性

`gt_data/results/final_wrench/oracle_vs_ekf_table.md` 明确写明 **3 cases × 5 noise realizations**，与 candidate FINAL 的默认值一致。对应 CSV 的 8D 行为：

| 方法 | Tip mm | 历史“RMS” mm | Force direction ° | 历史 NRMSE-F |
|---|---:|---:|---:|---:|
| 保存 Oracle | 0.0462723 | 0.0230593 | 2.0623285 | 0.0643787 |
| 保存 EKF 3-IMU | 0.9700590 | 0.4518987 | 61.8629002 | 1.8107448 |
| 论文 Oracle | 0.14 | 0.07 | 0.5 | 0.012 |
| 论文 EKF 3-IMU | 1.05 | 0.41 | 44.2 | 1.370 |

另一个 `gt_data/results/oracle_vs_ekf_results.json` 保存配置为 gamma=10、alpha=1、sigma=.5°、steps=30；8D 三 IMU 共 250 runs，均值 direction=94.2928512°、NRMSE-F=7.3560858。这是不同结果族，不能拿其配置/统计来解释上面的 3×5 表。

final_wrench 文件修改时间为 2026-04-23，早于 Git 中 2026-04-25 的 tendon 修正。文件时间仅提供时序线索，并非代码版本证明。原始 table CSV 没有输入/代码哈希；重跑当前 bc4d13a 原 API 得到不同数值时，必须区分输入恢复、提交变化和稿件数字来源三个问题。

## 本次只读校验的 SHA-256

| 文件（crt 相对路径；FINAL CSV 同样相对 crt） | SHA-256 |
|---|---|
| `compare_oracle_vs_ekf_wrench_estimation.py` | `3954c3ff97151b722c2329a72cd31f4fe8ce0d7fcf1f285fc04f41a28b59cd9e` |
| `evaluate_kirchhoff_shape_estimation.py` | `90f89487f18a23d0cfb65919451378e33d86e5187b04d4ae5ed8ee8a46cb1826` |
| `virtual_work.py` | `10dd7179bd42013e6303a31cd7291ae17cfe638ab40cd950c71811aeae2a8cb5` |
| `sweep_ekf_wrench_convergence.py` | `378c94027e02839b6dd9bfd963eb1a496e9f2885447c2da8291229e1089cd855` |
| `gt_data/kirchhoff_gt_dataset.npz` | `ae8ec9a3629a136e02b7919bb47eabe537e80e448457295fe9465494a608ba3a` |
| `gt_data/kirchhoff_imu_3imu.npz` | `ef709d85eb766e58dd8cf1358fd089a1623931c576ed857ecfcfdd7f026fd6e3` |
| `gt_data_seq/kirchhoff_imu_seq_3imu.npz` | `5772c840b83a99e08c3ffe82d89462567c04c9a18609acd693e8c44d04398338` |
| `gt_data/results/final_wrench/oracle_vs_ekf_table.csv` | `6a6cdb14cd15e6dbf1214064e5b86f19b5f31e071ac81ce5183e1a97b1523e64` |

建议实际诊断清单补充 Python/NumPy/SciPy 版本、命令参数、case/noise ID、实际传感器位置、数据 schema、源文件哈希、耗时、状态及显式指标公式。运行用 `python -B` 或 `sys.dont_write_bytecode=True`，将输出和 Matplotlib 缓存放在旧工作区；避免导入会 `os.chdir(CRT)` 的 FINAL，也不要执行会把输出写回 NEW 的 main。
