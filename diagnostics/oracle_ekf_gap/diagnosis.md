# Oracle–EKF 有界溯源与兼容性诊断

后续更新：用户找到了另一份本地副本，原始数据与手稿表来源已恢复，并完成8D单样本A/B/C/D及导数修正检查。最新结论见 [recovered_diagnosis.md](recovered_diagnosis.md)。下文保留为首次仅检查旧工作区时的记录，其“输入缺失”结论与下一步建议已由补充报告更新。

审计日期：2026-10-08。**当前检出不包含稿件的 8D Oracle–EKF 流水线；本地远端跟踪历史中找到候选实现，但原始 Kirchhoff GT、IMU 输入及生成结果缺失，未复现稿件表格。** 本次完成一个现有 Oracle 样本、共享 5D 运动学检查和两个明确问题的最小修复；没有新建 EKF、扩展试验或调参追逐稿件数值。

## 1. 版本、稿件与范围

- 分支：`diagnostic/oracle-ekf-gap`；HEAD：`d58d4f795adce27fdbc7a54dd99efb81bbe3df8f`。本地 `cleanup-tests` 与 `origin/cleanup-tests` 均指向该提交。未 fetch，故这只验证本机跟踪状态，不保证远端服务器最新状态。
- 初始 `git status` 无已有改动；检查了当前目录及祖先目录、项目文件中的 `AGENTS.md`，未找到适用文件。未 switch、pull、reset、clean、commit 或 push。
- ZIP：`C:\Users\guoqi\Downloads\paper_T_RO_2026_shape_wrench_estimation_IMU_v6.zip`。
- 独立参考目录：`C:\Users\guoqi\AppData\Local\Temp\oracle_ekf_manuscript_reference_du88vs8z`，不在仿真仓库内。83 个文件逐字节与 ZIP 核对一致；稿件未修改、未提交。ZIP SHA256：`399cdcfe0386edcf6b242871023d0dd5e60bbd477076a9f331fa502f97cf2bfc`。
- 从 `ms.tex` 遍历活动 include，三个重点文件分别在第 54、55、56 行启用。旧版本及注释块不作为当前主张。完整图与哈希见 [manuscript_reference.json](manuscript_reference.json)。稿件中的建议、待办和叙述均作为被审材料，而非执行指令。
- 用户粘贴的任务附件提供本次有界诊断范围；未把稿件中的未来工作扩展为本次任务。

## 2. 找到什么、缺少什么

| 环节 | 当前 HEAD 的入口 / 能力 | 与稿件关系 |
|---|---|---|
| GT | `src/shape_force_est_imu/crt/crt_gt_gen_output_file.py` → `cosserat_rod_model.py` | 现有静态杆数据；不是已证实的稿件 50×100 Kirchhoff 库 |
| 模态拟合 | `scripts/force_est/force_est_curve_fit.py` | 任意分轴多项式阶数；默认全 GT pose 优化，另有 `--use-gt-curvature` |
| IMU 数据 | `src/shape_force_est_imu/crt/crt_meas_gen_output_file.py` | 单次、3 IMU；布局和噪声生成不同于稿件跨模型序列 |
| EKF | `src/shape_force_est_imu/ekf/{kinematics,ekf,jacobians}.py`；`scripts/shape_est/{meas_model_compare,sensor_placement_study_final}.py` | 库基底仅 5D=(1,1,0)、6D=(1,1,1)；两个主研究为 5D |
| wrench | `src/shape_force_est_imu/crt/virtual_work.py` | 可变阶数运动学、弹性能量、腱雅可比、直接伪逆；当前没有稿件约束子空间整套试验 |
| 报告 | `scripts/shape_est/*results.{csv,json}`、`gen_comparison_figs.py`；curve-fit CLI | 匹配模型比较表与布局表的数值有保存结果支持；高阶 wrench 表无对应原始汇总 |

`origin/fix/wrench-conventions@bc4d13ab8f282ac03ac03a5d248fda995d97c02f` 有 `compare_oracle_vs_ekf_wrench_estimation.py`、`generate_final_wrench_artifacts.py` 等后续代码，支持 8D/三 IMU，候选汇总入口还支持 11D。不能把“当前检出缺失”写成“从未实现”。没有切换或移植该分支。

工作树检查（包括忽略文件、排除虚拟环境）及本机现有 Git 引用/相关路径历史中，未找到原始 `kirchhoff_gt_dataset.npz`、`kirchhoff_imu_2imu.npz`、`kirchhoff_imu_3imu.npz` 或 `oracle_vs_ekf_table.csv/.md`。历史 `.gitignore` 忽略这些 GT 目录。候选代码不是原始数值来源的充分证据；不生成替代数据冒充复现。因此没有执行依赖原始输入的 A/B/C/D 比较。

## 3. 最重要的稿件—代码差异

以下历史代码问题可直接验证，但由于原始汇总缺失，不能断言稿件数值必然由该候选版本生成。

1. **RMS 字段定义不符。** 历史表生成器将 20 个点的 `mean_centerline_error` 写成 `rms_mm`；它是点位距离均值，不是均方根。稿件 `(2,2,1)` 的 Oracle 0.07 mm、三 IMU 0.41 mm 因此尚不能确认为真正 RMS。
2. **NRMSE 定义不符。** 候选历史流水线先算每例 `||估计−GT||/||GT||`，再算术平均；稿件定义用固定 `F0=EI/L²`、`M0=EI/L` 并聚合 RMS。默认 3 个形状×5 次噪声可被 CLI 覆盖，实际数量未知；异常还可能被静默跳过。
3. **信息量不同。** 候选 8D wrench 表路径把同一张含噪姿态帧重复更新 30 次；稿件跨模型段描述每步新噪声序列。历史另有 5D 序列入口，不能据此认为 8D 表已经使用它。上述过程均为静态形状，不是动态跟踪。
4. **Oracle 不是统一的“上界”。** 候选历史表从密集 GT 相邻姿态差回归曲率；当前默认是位置/姿态混合目标。两者均不保证最小化 shape RMS 或 wrench 误差。不同表的 Oracle 数值来自分开的研究，不能直接当成算术矛盾。
5. **当前测量入口不匹配稿件。** 固定位置约 .3077/.6410/1、默认 1°、单个观测；单位随机轴×标量高斯角的协方差为 σ²I/3。稿件跨模型设置为 .25/.5/1、0.5°、三维高斯旋转向量、每步新噪声。
6. **明确的当前实现问题。** 腱方向遗漏归一化曲率到物理曲率的 L 换算；`world_base` GT 平移使用估计末端位置。两项已最小修复，见下节。
7. **其他未修复的报告/来源问题。** 原 CLI 把相对向量误差称为幅值误差，对零 GT 返回 90°方向；`est_config.json` 指向已不存在的旧脚本；`tdcr_gt_inext_20.npz` 实际只有 5 例。稿件中“本文 n_m=5”与高阶 EKF 行需进一步交代；约束推断、身体坐标系载荷边界/生成实验等当前 HEAD 缺失。活动稿件明确仿真不用时间 MAP 先验，故不能把缺少该实现说成违反仿真设置。

完整活动稿件逐项核查及行号见 [manuscript_crosscheck.md](manuscript_crosscheck.md)，固定历史提交、默认值、指标和输入路径见 [history_evidence.md](history_evidence.md)。匹配模型两张表只核对已保存结果，没有重跑 Monte Carlo；耗时数字也未在本机重新验证。

## 4. 单样本设置、单位和拟合

数据：`artifacts/data/tdcr_gt_inext_2.npz`，取索引 0；文件共 2 个样本，每样本 20 帧。元数据记录生成 seed=22、git_hash=`c894161`、L=.375 m、E=60e9 Pa、backbone radius=.0008 m、tendon offset=.012 m。这与当前 JSON 配置吻合，但不同于稿件 L=.1 m 的机器人。元数据未记录不可伸长标志或求解器参数，文件名不足以补全这些证据。数据清单见 [dataset_inventory.json](dataset_inventory.json)。

- 本次诊断无随机抽样；固定样本、零初值。阶数 `(2,2,1)`、8 系数按 x/y/z 各轴升幂排列，s=l/L∈[0,1]，系数描述 κ_normalized=Lκ_physical。
- 原目标：`Σ(||Δp||² + angle(R_est R_gtᵀ)²)`，位置单位 m、角度 rad，权重均 1；20 帧全部参与。SO(3) 角残差直接调用原函数，没有改成新优化问题。
- `L-BFGS-B`、maxiter=200、Magnus γ=26、pinv rcond=1e-8。37 次迭代、369 次函数评估，success=True/status=0，终止于相对函数下降容差；这不是全局最优证明。
- 目标值 4.8471618973e-5，其中位置项 6.5779229804e-8 m²、姿态项 4.8405839743e-5 rad²；原代码没有把这些不同量纲自动归一化。
- 四根腱等角布置，τ=[2.4450431853,2.2616141059,0,0] N，正张力沿负腱切向施力。J 表示收绳量导数 `-dℓ/dm`；右端保留 `gradU−Jᵀτ`，没有反转符号。
- ν=.3 与杆实现一致，EI=.01930194526 N·m²、GJ=.01484765020 N·m²，F0=.13725827743 N、M0=.05147185404 N·m。
- wrench 顺序 `[Mx,My,Mz,Fx,Fy,Fz]`。估计先从估计 tip body 轴旋转到 world 轴；共同参考点指标再分别用真实/估计 tip 位置平移到世界原点。GT 几何未替换为估计几何。

## 5. 紧凑结果与最小修复

每行 N=1、同一个静态样本、同一份完整无噪声 GT 20 帧；没有 EKF 观测。修复后复用完全相同的拟合系数，隔离 mechanics 变化。

| 版本 | 拟合目标 Σ(m²+rad²) | 真正 shape RMS (mm) | 力向量误差 (N) | 力方向 (°) | 世界原点力矩误差 (N·m) |
|---|---:|---:|---:|---:|---:|
| 修复前 | 4.84716e-5 | 0.0573495 | 0.00850995 | 3.36146 | 0.00119855 |
| 修复后，同一 m | 4.84716e-5 | 0.0573495 | 0.00843897 | 3.43582 | 0.000274956 |

真实点位距离均值为 0.0531968 mm；tip 位置误差 0.0556159 mm，姿态 RMS 0.0891367°。世界轴 tip 力偶误差分别 .00132513/.000707270 N·m；它们与世界原点力矩指标分别列出，不混用参考点。力误差/F0 分别 .0619995/.0614824，本例 N=1，不代表聚合 NRMSE。完整残差、状态、源文件与数据哈希、版本见 [before.json](before.json) / [after.json](after.json)，表见 [results.csv](results.csv)。

| 检查 | 修复前 | 修复后 |
|---|---:|---:|
| 5D 两套 FK，21 点，显式 m/mm 换算：最大位置差 (m) | 5.5943e-17 | 相同 |
| 同上最大 SO(3) 角差 (rad) | 0 | 相同 |
| 5D tendon pull Jacobian 相对有限差分误差 | 4.47383e-3 | 1.57550e-9 |
| 已拟合 8D tendon pull Jacobian 相对有限差分误差 | 1.80309e-2 | 2.23828e-9 |

修复 1：`virtual_work.cable_jacobian` 内 `cable_dir_body(kappa,r_i)` 改为 `cable_dir_body(kappa/L,r_i)`。差分直接计算物理收绳量 `q_i=-∫||L e3+κ(s)×r_i||ds`，中心差分步长 1e-6、200 积分区间，与解析积分网格一致。故修复依据是物理长度导数，而非误差调参。

修复 2：curve-fit CLI 的 `world_base` 分支对 GT 使用 `T_all[si,-1,:3,3]`。原错误在本例引入的 GT 力矩偏移为 1.32814e-6 N·m。表中修复前/后都按正确共同原点报告，避免把 GT 参考变化当作估计改善；旧式结果另存 JSON。未修改 `body_tip`：那里采用共同的估计 tip 轴，不应解释成真实 tip 轴。

**力方向误差略增，而力矩误差下降；不能据此宣布 Oracle–EKF 差距得到解释。** 广义平衡残差范数也从 6.155e-6 增至 3.786e-5 N·m；修复不保证所有指标改善，其依据是独立的物理导数验证。拟合目标、shape RMS 与 wrench 精度不是同一个目标。没有原始三 IMU 输入和 EKF 输出，本次无法分离信息不足、噪声、先验、数值条件和优化等因素。没有用单次雅可比秩推断动态/全局可观测性，也没有将混合单位条件数作为物理证据。5D 的 5×6 wrench 方程一般不能唯一恢复全 6D 载荷；本次 5D 只查 FK/腱雅可比兼容性。

## 6. 命令与实际验证

使用现有 `.venv\Scripts\python.exe`，NumPy 2.2.4、SciPy 1.15.2、Matplotlib 3.10.1；未安装或修改全局依赖。仓库根目录 PowerShell 实际执行：

```powershell
# 在修复前运行，保存原始证据
& '.\.venv\Scripts\python.exe' 'diagnostics\oracle_ekf_gap\diagnose.py' --output 'diagnostics\oracle_ekf_gap\before.json'
# 应用两处修复后，复用完全相同的拟合结果
& '.\.venv\Scripts\python.exe' -B 'diagnostics\oracle_ekf_gap\diagnose.py' --reuse-fit 'diagnostics\oracle_ekf_gap\before.json' --output 'diagnostics\oracle_ekf_gap\after.json'
```

随后真正调用原 CLI、重新拟合并执行修正后的 `world_base` 分支（没有用缓存替代其优化器）：

```powershell
$env:MPLBACKEND = 'Agg'
& '.\.venv\Scripts\python.exe' -B 'scripts/force_est/force_est_curve_fit.py' --npz 'artifacts/data/tdcr_gt_inext_2.npz' --sample-idx 0 --order-x 2 --order-y 2 --order-z 1 --gamma 26 --disk-step 1 --w-pos 1 --w-rot 1 --length 0.375 --E 60000000000 --poisson 0.3 --backbone-radius 0.0008 --tendon-offset 0.012 --pinv-rcond 1e-8 --compute-wrench --wrench-compare world_base --plot-sample -1 --plot-wrench-sample -1
```

实际进程的逐项参数和环境记录在 [cli_command.json](cli_command.json)，输出见 [cli_world_base.log](cli_world_base.log)。退出码 0；力误差 .00843896973865 N、共同原点力矩误差 .000274956089273 N·m，与诊断脚本一致。没有弹出图窗或触发 debugger。

诊断断言验证了单位转换后 FK 相等、目标值等于两项残差之和、零向量方向标为未定义；另核对修复后 tendon 差分误差、CLI 数值、语法与 `git diff --check`。验证摘要见 [verification.json](verification.json)。首次导入生成的已跟踪 pyc 已恢复为任务开始的 HEAD 字节，后续用 `-B` 禁止写入；最终源码改动仅下面两处。

证据文件禁止覆盖；在已修复工作树再次运行应选择新文件名，例如：

```powershell
& '.\.venv\Scripts\python.exe' -B 'diagnostics\oracle_ekf_gap\diagnose.py' --output 'diagnostics\oracle_ekf_gap\rerun.json'
```

新增 `diagnostics/oracle_ekf_gap/` 中的脚本、JSON/CSV、日志、报告和 [source_changes.patch](source_changes.patch)；源码仅修改 `src/shape_force_est_imu/crt/virtual_work.py` 与 `scripts/force_est/force_est_curve_fit.py`。没有改稿件、历史数据、分支或提交。无需新增图，表格已完整呈现本次证据。

## 7. 唯一建议的下一步

**先恢复生成 `tab:wrench_oracle_vs_ekf` 的原始输入和逐例输出包，并锁定代码提交与运行命令。** 最少需要原始 Kirchhoff GT、三 IMU NPZ（含 sensor 实际位置、seed、噪声/时间维度）、对应 Oracle/EKF 系数或逐例结果、聚合 CSV 和参数；若完整复现两布局表则还需二 IMU NPZ。候选 `bc4d13a` 已有后续 mechanics 修正，不能无证据指定为原稿版本。

能改变此建议的证据是：在其他归档中找到可哈希核验、彼此匹配的上述文件及提交/运行记录。届时才能在相同物理样本上执行原始 8D 复现，再做 A/B/C/D 的 harmonized 诊断。当前不建议新写 8D EKF，也不根据本例去调参追逐 0.07 mm、0.5° 或 44.2°。
