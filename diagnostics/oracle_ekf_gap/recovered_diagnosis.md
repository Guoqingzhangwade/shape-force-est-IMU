# 找回原始副本后的 Oracle–EKF 单样本诊断

日期：2026-10-08。**已找到手稿结果的保存来源及原始 GT/IMU 数据，并完成原 8D、三 IMU 流程的单样本运行与 A/B/C/D 对照。** 此前“输入尚未恢复”的阻塞已解除；没有重新运行完整聚合表，也没有把单例结果当成稿件均值。

## 1. 仓库和来源

VS Code 的 `windowsState.lastActiveWindow.folder` 记录指向 `C:\Users\guoqi\OneDrive - stevens.edu\Documents\GitHub\shape-force-est-IMU`。直接窗口读取失败：`Computer Use native pipe is unavailable`；路径识别来自本地工作区记录，随后由实际文件和 Git 状态确认。

新副本当前为 `fix/wrench-conventions@bc4d13ab8f282ac03ac03a5d248fda995d97c02f`。原有三个修改文件：`cosserat_rod_model.py`、`evaluate_kirchhoff_wrench_constrained.py`、`generate_kirchhoff_gt_dataset_constrained.py`；另有未跟踪嵌套 `crt/src/`。未修改、合并或清理该副本；实验只读导入原文件，禁止写 pyc，所有输出放在原可写仓库的 `diagnostics/oracle_ekf_gap/`。相关源码/输入哈希及 Git status 前后相同。

原工作区继续保留 `diagnostic/oracle-ekf-gap@d58d4f7` 和前一阶段的两处修复。稿件仍为仓库外临时参考目录，内容不变。本次检查了新路径祖先及项目中的 AGENTS.md，未找到适用文件。

## 2. 手稿数字已经能追溯

确切匹配的文件位于新副本：

```text
src/shape_force_est_imu/crt/gt_data/results/
  committee_20260426_1647/final_wrench_committee/oracle_vs_ekf_table.csv
```

该 CSV 的 12 行在手稿显示精度下全部吻合。对应 MD 记录 **20 cases × 3 noise realizations**；`run_log.txt` 记录 `generate_final_wrench_artifacts.py --n-cases-ekf 20 --n-noise 3 --max-cases-oracle 50 --max-cases-constrained 50`。独立 Oracle 阶数表实际使用另一个 10 例 TDCR 数据集，不是该 20 例样本集。

| (2,2,1) / 8D | CSV 的“rms_mm” | 力方向 (°) | CSV 的“NRMSE_F” |
|---|---:|---:|---:|
| Oracle | .06726711650585192 | .5299539460927635 | .01153749914389534 |
| EKF 3-IMU | .41448609071486736 | 44.24148754643709 | 1.3695467006228352 |

因此 0.07、0.41、0.5°、44.2° 均有直接的保存数值来源。但是指标含义仍需纠正：

- `rms_mm` 来自 20 个点位距离的**算术平均**，不是 RMS。点索引为 `round(linspace(0,M-1,20))`，见表生成器第 148–160 行。
- `NRMSE_F/M` 为各例相对向量误差 `||估计−GT||/||GT||` 的平均，既未使用稿件固定特征尺度 F0/M0，也未进行平方平均再开方。
- 该高阶 wrench 表使用静态含噪帧，反复更新同一观测。稿件的跨模型几何结果则来自另一套逐步新噪声序列。两者都是静态形状试验。

同名 `gt_data/results/final_wrench/oracle_vs_ekf_table.csv` 来自另一轮 3×5 实验，其 8D 三 IMU 力方向为 61.8629°；不能按文件名混用。

还找到并核对了跨模型几何、独立 Oracle 扫描、force-only、身体坐标约束及可观测性表。详见 [recovered_manuscript_crosscheck.md](recovered_manuscript_crosscheck.md)。此前 Fy/Fz 标签疑点也得到解释：保存结果分别记录最弱非零奇异方向的 dominant component=Fy 与 null direction 的 dominant component=Fz；手稿应交代两种含义。

保存 CSV、日志和当前源码共同支持来源关联，但缺少生成时完整源码哈希/干净工作树 manifest，不能把当前 HEAD 自动当作每张历史表的精确生成版本。

## 3. 本次固定样本和配置

输入来自 `crt/gt_data/kirchhoff_gt_dataset.npz` 与 `kirchhoff_imu_3imu.npz`，选 **case index/id=0、noise realization=0**：

- GT：50 个形状×100 点，L=.1 m、E=60 GPa、ν=.3、backbone radius=.0005 m、tendon offset=.008 m、4 腱、不可伸长，GT seed123；元数据生成 hash=d58d4f7。
- IMU：5 次静态 noise realization，0.5° 左乘 Cartesian Gaussian rotation vector，seed1234；本次不重新抽取噪声。
- 标称位置 .25/.5/1；实际采样位置 `[25/99,50/99,1]`，与 GT 的索引 25/50/99 完全一致。本次与手稿表生成器一样使用 actual_s。独立 COMPARE CLI 使用名义位置，属于另一入口选择。
- 阶数 `(2,2,1)`，8 维、x/y/z 升幂分块；normalized s=l/L 和 κ=Lκ_physical；γ=10、α=1、30 updates。
- 原 EKF 零初始化，P0=diag(1,.5,1,.5,.25,.25,.25,.25)；Q=diag(1e-10,1e-12,1e-10,1e-12,1e-10,1e-12,1e-12,1e-12)。同一含噪三姿态反复使用，不产生30个独立观测。
- 所有方法使用相同 `wrench_from_modal`、EI=.00294524311274 N·m²、GJ=.00226557162518 N·m²、同一 GT tension、已含正确 L 尺度的 pull Jacobian、pinv rcond=1e-8。没有套用前一阶段 .375 m 机器人的参数。

完整数据/源码哈希、状态、配置、种子及优化器结局在 [recovered_run/results.json](recovered_run/results.json)。

## 4. A/B/C/D 的信息量与结果

A 是原稿候选代码的密集 Oracle：100 个无噪声 GT 姿态的相邻 SO(3) 差分得到 normalized curvature，再逐轴线性最小二乘。位置没有参与拟合；没有非线性初值或迭代收敛概念。它与前一阶段的 pose-fit 目标不同。

B/C 使用 SO(3) 残差 `Log(R_obs R_predᵀ)`，除以同一 σ；无先验，三个固定且不使用 GT 系数的初值：零、seed1234生成的 .01 与 .1 倍 Gaussian 扰动。SciPy TRF 数值雅可比、max_nfev120、xtol/ftol/gtol=1e-10；六次拟合都报告成功，分别得到相同残差到数值精度。成功不能证明全局唯一解。

D 直接调用原 `run_ekf_for_order`，保留原初始化、协方差、更新、解析雅可比、30步和输入。先保存原 A/D 状态，再进行离线拟合及任何导数修正检查。

下表每行 **N=1**；shape 为全部100点的真正 RMS。观测残差为9维旋转向量拼接后的欧氏范数，单位rad，不等同于密集Oracle的曲率LS目标。

| 方法 / 信息源 | 本方法观测残差范数 (rad) | shape RMS (mm) | 力误差 (N) | 力方向 (°) | 世界原点力矩误差 (N·m) |
|---|---:|---:|---:|---:|---:|
| A：密集无噪声100姿态；残差列仅在3个无噪声IMU位置评价 | .00168656 | .0143924 | .000989835 | .200049 | .0000332250 |
| B：单个无噪声三姿态 | .00118067 | .0202995 | .0322491 | 5.38576 | .000122559 |
| C：单个含噪声三姿态，与D相同 | .00617800 | .661900 | .778697 | 94.0307 | .00440233 |
| D：同一含噪声三姿态重复30次，原EKF | .00838666 | .554543 | .649148 | 89.6631 | .00362692 |

A 的实际拟合曲率残差另存 JSON，不能与 B/C 的 orientation objective 按同一列比较。所有帧都使用世界轴；估计与 GT 力矩分别从各自 tip 位置平移到共同世界原点，GT 几何始终保持真实值。世界轴 tip 力偶指标也单独保存。接近零的估计或 GT 向量方向输出未定义，当前样本实际非零。

`F0=EI/L²=.294524311274 N`；力误差/F0 分别 .00336079、.109495、2.64391、2.20406。它们是单例固定尺度误差，不是稿件历史“NRMSE”字段。该例约90°不能用于否定或代替20×3聚合的44.2°。

## 5. 明确的指数导数错误及最小修正检查

原 `compare_oracle_vs_ekf_wrench_estimation.py:621` 使用：

```python
dePsi_dm[k][:, :, i] = ePsi_k @ (dP + 0.5 * (Psi_k @ dP - dP @ Psi_k))
```

在同一 Oracle 状态、实际IMU位置、同一 residual 与 additive coefficient perturbation 下，用中心差分 eps1e-6 直接核对嵌套 `H_and_r_general`。原 H 的相对 Frobenius 误差 **.103601817**。将该表达式替换为现有 SciPy `expm_frechet(Psi_k,dP,compute_expm=False)` 后，相对误差为 **1.52991e-10**。

修正只在内存中的第二个原模块实施；**新副本源码没有修改**。完整可审核补丁见 [recovered_run/jacobian_correction.patch](recovered_run/jacobian_correction.patch)，检查脚本与原/修正结果见 [recovered_run/jacobian_correction.json](recovered_run/jacobian_correction.json)。它保留相同滤波器，未新增估计器或调参。

| D 同样本 / 同配置 | 原实现 | 仅修正导数 |
|---|---:|---:|
| shape RMS (mm) | .554543 | .552549 |
| 力误差 (N) | .649148 | .646810 |
| 力方向 (°) | 89.6631 | 89.2880 |
| 世界原点力矩误差 (N·m) | .00362692 | .00366483 |

导数错误确实存在，但修正后本例的大 wrench 误差基本保留；不能将原稿差距归因于它。没有在恢复副本上应用生产修复，避免覆盖已有工作；补丁已具体化并经过验证。

## 6. 能说明什么、不能说明什么

- A→B：B在三个无噪声传感器位置的姿态残差更低，shape RMS从.01439增至.02030 mm，wrench却明显变差。即使几何误差很小，也不保证准确wrench；数据与模型离散/表示残差仍存在。
- B→C：同一个稀疏 orientation fit 加入这份保存噪声后，wrench 明显变差；这支持噪声和模态误差放大的局部解释，未证明三IMU普遍不足或全局不可观测。
- C 的 noisy residual 小于 D，但物理 wrench 更差；因此继续降低观测拟合误差不能直接充当提高载荷精度的策略。原EKF先验/有限更新可能产生不同解，本次不分离这些贡献。
- 原 EKF 导数已独立核实有错；仅修正导数不足以消除本例误差。
- 本次检查 general FK 与 mechanics FK 的8D变换一致至数值精度。没有把默认库L=100误判为米/毫米错误。
- `A(m)w=b(m,τ)` 两侧都随shape变化；JSON同时记录 common world/base map 下的 `δb`、`δA·w_oracle` 及两者合成扰动。未只看b，也未用未缩放混合单位条件数解释物理敏感性。
- 单例、三个初值与一次导数检查都不能建立全局/动态可观测性，或代表整个20×3样本集。

## 7. 命令、文件和检查

原工作区根目录 PowerShell 实际执行，使用已有 Python3.10.11/.venv、NumPy2.2.4、SciPy1.15.2：

```powershell
& '.\.venv\Scripts\python.exe' -B 'diagnostics\oracle_ekf_gap\recovered_run\diagnose_recovered.py'
& '.\.venv\Scripts\python.exe' -B 'diagnostics\oracle_ekf_gap\recovered_run\check_jacobian.py'
```

精确 argv/运行边界见 JSON。两个脚本均禁止覆盖已有结果；再次运行主脚本应传新的 `--output-dir`。输出中没有交互图窗，没有生成新GT/IMU，没有换分支或提交。

`diagnose_recovered.py` 初次输出的“历史mean20”字段曾用floor索引，后按原表生成器的round索引重算该字段；拟合状态、力、力矩和100点RMS不变，原始输出保存在 `results_before_metric_index_fix.json/.csv`。这是诊断指标选点修正，未改历史数据或运行更大的试验。

实际检查：匹配保存表与手稿舍入、输入metadata/硬编码mechanics一致、真实IMU索引一致、general/mechanics FK一致、三个初值的收敛与残差、原/修正H的有限差分、两次独立原D运行状态一致、恢复副本相关源码/数据SHA256及Git status不变、诊断语法和原工作区diff格式检查。结果见 [recovered_run/verification.json](recovered_run/verification.json)。

本轮新增/更新只在原工作区诊断目录：两个脚本、单例状态/JSON/CSV、导数补丁/检查、两份补充来源报告、本报告与验证。原报告顶端补充后续链接。没有改稿件或新副本，也没有再次改动原工作区的生产源码。

## 8. 唯一建议的下一步

**先修正手稿的指标名称/公式及高阶 wrench 表的观测与样本说明，并引用已找到的明确结果目录。** 本次已证明0.07/.41列不是RMS，1.370不是稿件定义的固定尺度聚合NRMSE；若不先厘清含义，继续解释“shape RMS很小但wrench很差”会混用指标和信息预算。导数补丁保留为下一次代码工作的具体输入。

能改变这一建议的证据是另一个有完整源码哈希与运行配置的结果包，证明同样稿件数字实际由真正RMS/固定尺度NRMSE及逐步独立观测生成。当前精确匹配CSV、日志与生成器支持上述修订。本次止于这条证据链，不扩展参数扫描、滤波器或手稿范围。
