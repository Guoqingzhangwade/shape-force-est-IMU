# 两份本地仓库与 GitHub 的核对

日期：2026-10-08。GitHub 分支通过实际执行只读 `git ls-remote --heads origin` 核对，未使用几个月前的 remote-tracking 缓存作为线上现状。完整提交编号及源码快照哈希见 [reconciliation.json](reconciliation.json)。

| 位置 | 当前分支/HEAD | 与实际线上比较 |
|---|---|---|
| 桌面旧副本 | diagnostic/oracle-ekf-gap / d58d4f7 | 基点与线上cleanup-tests相同；有本次诊断改动/结果 |
| 教育OneDrive/GitHub目录新副本 | fix/wrench-conventions / bc4d13a | HEAD与线上同名分支完全一致；另有3份未提交源码改动 |

d58d4f7是bc4d13a的祖先，两者之间有28个追加提交。新副本的3份modified文件有实际功能：tip-body载荷求解、local transverse/moment数据生成、评估及零方向NaN处理。在本机可见提交中未找到这些新增功能；当前线上同名分支亦只有bc4d13a。

## 已保存的内容

- `modified_source_snapshot/`：新副本3个modified源码文件，保留原文并逐个SHA256验证。
- `new_repo_uncommitted.patch`：相对HEAD的完整源码diff。
- `reconciliation.json`：实时线上提交编号、状态、哈希和检查边界。

**这只是源码保全快照，不是完整备份。** 它位于旧仓库内，删除旧仓库也会删除此快照。被忽略的GT/IMU/结果、未跟踪嵌套输出、Git历史及另一设备上的工作尚未在独立位置归档。整仓库递归元数据扫描已停止，没有据此声称完成了全数据备份。

## 处理建议

目前两份都保留。先在两份工作目录之外做可恢复的完整归档，包含Git历史、modified/untracked/ignored文件；虚拟环境可不保存。再检查另一台设备的HEAD、分支、modified/untracked文件及未推送提交，避免把GitHub误当成另一台设备的全部工作。

新副本更适合作为后续研究的基准，因其包含更后的提交和手稿数据/结果。旧副本有本次诊断修复和报告，应逐项移入或归档；不要整个旧src覆盖新src。已有代码审查和验证后，再将需要保留的源码变成明确提交并与GitHub同步。未完成保全和对照前，不删除任一副本。

只有在完整归档已验证可恢复、另一设备的独有工作已确认、新主工作目录包含需要的源码/数据/诊断且检查通过后，旧重复目录才可以删除。两台设备各保留一份工作仓库，通过Git提交同步代码，数据和实验结果另行备份；不要依赖“modified文件日期”决定覆盖方向。
