# 当前稿件与当前代码交叉核查

代码差异与行号以审计开始时的 HEAD `d58d4f7` 为准；本次后续修复的两处源码及前后验证见 [diagnosis.md](diagnosis.md)，不应把下文修复前的问题描述当作修复后现状。

本文件记录只读证据核查，不修改稿件或仿真源码。结论区分“数值有保存结果支持”“当前工作树未找到实现”“配置或指标不一致”与“缺少来源，暂不能判断”。当前 HEAD 缺失不表示功能从未实现；其他审计分支已找到后续历史实现，历史提交的实现与指标问题由历史审计单独记录，不混同当前可执行版本。

稿件参考目录：`C:\Users\guoqi\AppData\Local\Temp\oracle_ekf_manuscript_reference_du88vs8z`。该目录位于仓库之外，保留为只读参考；本文不复制稿件内容到仓库。以下稿件路径均相对于此目录，代码路径均相对于仓库根目录。行号针对本次检查的文件。

## 1. 活动稿件范围

由 `ms.tex` 的活动 `\input` 确认：

```text
ms.tex
├─ content/abstract                              :51
├─ content/intro_v2                              :53
├─ content/modeling_v2                           :54  ← 本次重点
├─ content/estimation_framework_formulation_v2    :55  ← 本次重点
├─ content/sim_study_v3                           :56  ← 本次重点
├─ content/conclusion                            :59
└─ content/appendix                              :60
```

`ms.tex:52,57–58` 的 checklist、experiments、discussion 被注释，不作为当前活动章节；未将旧版 modeling、estimation 或 sim_study 文件混入证据。活动文件内部也有旧稿：`content/sim_study_v3.tex:594–717` 的旧扳手基线、参数敏感度段落，以及 `content/estimation_framework_formulation_v2.tex:524–650` 的旧协方差/MAP 段落均被注释，不作为当前承诺。

## 2. 主张—证据—结论矩阵

| 活动稿件主张及位置 | 当前代码或保存结果证据 | 核查结论 |
|---|---|---|
| 10 个匹配模型形状、每形状 3 次噪声、每次 50 步；3 IMU 位于 .25/.5/1；0.5°、α=1；给定 Q、P0。`sim_study_v3.tex:28–68` | `scripts/shape_est/meas_model_comparison_results.json:2–26`；`sensor_placement_results.json:2–21`；`sensor_placement_study_final.py:68–77,278–280` | 保存配置一致。另保存 shape seed=123、noise seed base=1000、γ=10；这些是重现所需的额外配置。 |
| 四种测量模型的 RMSE、NIS、耗时表。`sim_study_v3.tex:85–88` | `scripts/shape_est/meas_model_comparison_results.csv:2–5` | 表中所有列与保存 CSV 在稿件舍入精度下一致；不是本次重新跑完整 Monte Carlo 得到。 |
| 每种 IMU 数量的最优位置及末态 RMSE。`sim_study_v3.tex:118–122` | `scripts/shape_est/sensor_best_by_count.csv:2–6` | 位置、均值、标准差与保存 CSV 一致。 |
| α∈{.1,.2,.5,.8,1,1.5,2,3} 扫描。`sim_study_v3.tex:66–68` | `meas_model_compare.py:1647–1665,1759–1780` 有扫描入口；当前文件清单未找到扫描输出 CSV/JSON | 实现存在；完整扫描曲线的原始数据来源未在当前树闭合。α=1 的配置和比较表有保存证据。 |
| Kirchhoff 几何/材料：L=.1 m、半径 .0005 m、E=60 GPa、ν=.3、4 根腱、偏置 .008 m、40 disks。`sim_study_v3.tex:202–216` | `crt_gt_gen_output_file.py:133–147`；`cosserat_rod_model.py:28–38,41–50` | 生成器默认几何和刚度公式支持该组参数；这不证明某个已保存数据集用它生成。 |
| 50 个有效构形、每形状 100 点、seed123、最多500次、剔除横向末端位移<1 mm。`sim_study_v3.tex:232–240` | `crt_gt_gen_output_file.py:108–112,145,149,194,244–249` 默认 n10/seed22/maxtries1000，采样点数与 rod.N 绑定，成功求解后直接保存 | 默认配置不同可用参数解释部分差异；独立100点采样和近直线拒绝逻辑当前未找到。当前 NPZ 清单没有50×100构形数据。 |
| 跨模型 IMU 布局 .5/1 或 .25/.5/1；0.5° 三维高斯旋转向量；每工况5条逐步独立噪声序列。`sim_study_v3.tex:255–277` | `crt_meas_gen_output_file.py:104,119,126,133–148` 固定 .307692/.641026/1、默认1°、每工况每传感器单个四元数、随机单位轴乘标量高斯角 | 当前独立测量生成器不实现所述序列实验；即使改成0.5°，其旋转向量协方差为 σ²I/3，不是 σ²I。匹配模型的三维高斯采样则一致，见 `sensor_placement_study_final.py:352`。 |
| 跨模型平均位置误差 .460/.349 mm，平均姿态误差 2.483°/2.313°。`sim_study_v3.tex:292` | 当前树未找到对应50工况×5序列结果和运行脚本 | 当前 HEAD 不能追溯这些结果；不据此断言结果错误。 |
| 丰富模态阶数下的端到端 SO(3) EKF 与扳手推断。`sim_study_v3.tex:470–481` | `src/shape_force_est_imu/ekf/kinematics.py:11–21` 只支持5d=(1,1,0)、6d=(1,1,1)；当前 SO(3) 比较/布局脚本固定5D | 一般阶数的力学与 oracle 拟合存在，但当前 HEAD 未找到用于表中7/8/11维状态的端到端 EKF 实现及结果。历史实现另核查。 |
| 约束子空间推断及身体坐标系载荷实验。`sim_study_v3.tex:513–523,534–580` | `virtual_work.py:221–224` 仅直接伪逆；`crt_gt_gen_output_file.py:39–56` 对世界坐标力分量置零；`cosserat_rod_model.py:219–226` 使用固定空间载荷边界 | 当前 HEAD 未找到约束估计、身体坐标系载荷边界或这些表的保存输出。moment-only 世界坐标力为零并不等于已实现稿件指定的身体坐标载荷生成及验证流程。 |
| 归一化弧长的腱切向量为 L e3+κ×r。`modeling_v2.tex:451–525` | `virtual_work.py:177–198` 使用 e3+κ×r；`cable_jacobian` 接收 L 但没有使用 | 明确的公式/实现差异。它改变非单位长度下有扭转的腱方向及载荷映射，不能仅靠输出单位换算消除。 |
| 对接近零的真实载荷不统计方向角。`sim_study_v3.tex:360` | `force_est_curve_fit.py:409–412` 分母加ε后无零载荷筛选 | 当前输出会把真实零向量统计为90°方向误差；与活动稿件定义不同。 |

表内代码短名的目录：`crt_gt_gen_output_file.py`、`crt_meas_gen_output_file.py`、`cosserat_rod_model.py`、`virtual_work.py` 位于 `src/shape_force_est_imu/crt/`；`force_est_curve_fit.py` 位于 `scripts/force_est/`；测量比较和传感器布局脚本位于 `scripts/shape_est/`。

## 3. 指标、单位和参数来源

- **匹配模型 RMSE 是系数空间指标。** `meas_model_compare.py:866,879` 与 `sensor_placement_study_final.py:315–323` 对模态系数差取均方根；“RMSE mean”先对每时刻系数 RMSE 沿时间平均，再跨试验汇总。不能把这里的 .00743 当作毫米或几何 RMS。
- **NIS 的 ± 含义需写清。** 稿件 SO(3) analytic 的 `9.24±4.16` 来自跨运行的“时间均值的均值”与“时间标准差的均值”；不是30个运行均值的标准差，后者约 .6567。见 `meas_model_compare.py:887–888,2028–2030` 和保存 CSV。数值本身与稿件一致。
- **归一化曲率。** `modeling_v2.tex:151–177` 定义 s=l/L、κ=Lκ物理，模态参数描述归一化曲率。当前 PoE 最后将位置乘L，导数亦乘L，能量梯度使用 EI/L、GJ/L（`virtual_work.py:98,159,201–208`），与该约定相容；腱切向量的L遗漏是独立问题。
- **扳手顺序和量纲。** 活动模型采用 `[moment; force]`（`modeling_v2.tex:540–562`）；代码 `virtual_work.py:13–17` 和 `force_est_curve_fit.py:397–400` 一致。稿件 F0=EI/L²、M0=EI/L（`sim_study_v3.tex:332–344`），当前报告代码亦如此（`force_est_curve_fit.py:385–388`）。对应 .1 m/.0005 m/60 GPa 时 F0≈.2945 N、M0≈.02945 N·m。
- **当前“Relative magnitude error”标签不准确。** `force_est_curve_fit.py:407–408,417–418` 用 `||估计向量−真实向量||/||真实向量||`，不是 `abs(||估计||−||真实||)/||真实||`。活动稿件的指标段没有给出相对幅值误差公式，因此不能未经来源核对就认定其 oracle 表幅值列沿用了该实现。
- **默认命令不等于稿件实验命令。** `meas_model_compare.py:1602–1608` 默认是 `full_compare`，稿件保存表对应 `formulation_compare`。另一方面 `scripts/config/gt_config.json:4–18` 是 n=2、L=.375 m、半径 .0008 m、腱偏置 .012 m、20 disks 的另一组机器人配置；不能当作稿件参数来源。
- **估计入口已过时。** `scripts/config/est_config.json:2` 仍引用当前不存在的 `scripts/curve/3d_curve_fit_from_crt_gt.py`；实际相关脚本是 `scripts/force_est/force_est_curve_fit.py`。

## 4. 重点扳手表的来源歧义

`tab:wrench_oracle_orders`（`sim_study_v3.tex:443–447`）与 `tab:wrench_oracle_vs_ekf`（`:470–481`）的 oracle 数值不同：

| 阶数 | 独立 oracle 阶数扫描：NRMSE-F / 力方向误差 | oracle-vs-EKF 表：NRMSE-F / 力方向误差 |
|---|---|---|
| (1,1,0) | .563 / 36.1° | .528 / 34.9° |
| (2,2,0) | .082 / 3.9° | .194 / 10.3° |
| (2,2,1) | .012 / .6° | .012 / .5° |
| (3,3,2) | .002 / .1° | .001 / .1° |

稿件 `:429–430` 明确说独立 oracle 扫描与后续 EKF 对比是不同研究，故这里不能直接定性为表格算术矛盾。需要每张表的提交、数据集、样本索引、拟合方式、积分精度和指标代码才能解释差异。

当前 oracle 工具有两种不同含义：`force_est_curve_fit.py:275–289` 可直接拟合由相邻姿态差得到的曲率（`:99–117`），或优化位置/姿态联合目标（`:45–51`，默认权重将 m² 与 rad² 相加）。后者不保证最小化表中的位置 RMS；前者也不等价于“真实模态状态”。当前未找到原表运行命令，不能断言哪一种生成了每一行。

`tab:wrench_oracle_vs_ekf` 的 Tip/RMS 在 caption 中声明单位为 mm（`sim_study_v3.tex:461`），但活动段落没有给出足够的聚合/schema信息来确认每个字段如何导出。历史脚本字段语义或转换错误需根据相应历史提交及CSV另外判断；不要拿当前匹配模型的系数 RMSE 定义替代该表定义。

其它需要澄清的稿件表述：

- `estimation_framework_formulation_v2.tex:41` 写“本文 n_m=5”，但后续有7/8/11维 EKF 行；阶数扩展后的 Q、P0、初值没有在活动实验设置中明确。
- `sim_study_v3.tex:413` 的五维行中 Fz observable fraction=.345，低于 Fy=.839，但“weakest comp.”写Fy；`:400`又说Fz跨阶数最弱。需要明确该列是否采用另一指标，而非直接认定列值错误。
- 广义载荷协方差及MAP公式确实在活动稿件中（`estimation_framework_formulation_v2.tex:333–449`），当前实现未找到；然而`:455–471`明确说明报告实验采用单位权重子空间最小二乘且没有时间先验。因此缺少MAP实现不证明实验违背了其明确的LS设置。

## 5. 实际 NPZ 清单摘要

本次直接读取数组 shape 与 meta，未凭文件名推断样本数。

| 文件（均位于 `artifacts/data/`） | 实际数组结构 | 关键元数据/限制 |
|---|---|---|
| `tdcr_gt_samples_100.npz` | T=(100,40,4,4) | L=.1，seed=null，2025年的保存数据；不是50×100稿件库。 |
| `tdcr_gt_samples_10.npz` | T=(10,40,4,4) | L=.1，seed22。 |
| `tdcr_gt_samples.npz` | T=(10,40,4,4) | L=.1，seed=null。 |
| `tdcr_gt_force_only.npz` | T=(10,40,4,4) | seed22；不是稿件50例约束数据集。 |
| `tdcr_gt_inext_20.npz` | **T=(5,40,4,4)** | 文件名20，但数组和meta均为5例，seed22。 |
| `tdcr_gt_inext_5.npz` | T=(5,40,4,4) | seed22。 |
| `tdcr_gt_inext_2.npz` | T=(2,20,4,4) | L=.375、半径.0008、腱偏置.012、seed22。 |
| `tdcr_gt_inplane_xz_c02_n3.npz` | T=(3,40,4,4) | seed22。 |
| 其它单工况 GT / zero / large 文件 | T=(1,40,4,4) | L=.1，seed22。 |
| `tdcr_meas_samples{,_10,_100}.npz` | q_meas=(10或100,3,4) | 1°、布局 .307692/.641026/1、张力噪声.02 N；没有时间/序列维度。 |
| `dataset.npz` | shapes=(100,4,4)、tensions=(100,4)、wrenches=(100,6) | 仅末端帧结构；不是完整100点骨架库，无meta。 |

多个旧 GT 元数据只记录长度和seed，缺少材料、半径、腱偏置、是否不可伸长、载荷范围及生成命令，不能从文件名或后来的类默认值补写这些来源。上述清单没有与稿件50例×100点相符的数据；后续历史版本若包含相应实现或输出，应以明确提交和实际schema另建关联。

## 6. 本核查的结论边界

匹配模型两张表有直接保存数值支持。跨模型几何、一般阶数EKF、可观测性与约束扳手结果在当前HEAD的实现和数据链条不完整；不能据当前树直接复现或验证其数值。腱切向量的长度因子、独立测量生成器的噪声/序列设置、零载荷方向处理则是可以在当前代码中直接定位的差异。后续的小规模诊断与历史审计应分别记录自身配置、样本数和来源，不将替代运行标成原表复现。
