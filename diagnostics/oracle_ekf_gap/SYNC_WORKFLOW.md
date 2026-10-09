# Working copy and synchronization

Use the Documents/GitHub checkout as the active working copy, on branch
`diagnostic/oracle-ekf-gap`. The migration starts at `bc4d13a` from
`fix/wrench-conventions`, which already contains the old desktop checkout's
`d58d4f7` and 28 subsequent commits. No older branch needs to be pushed first.

The branch preserves the three previously uncommitted body-frame load changes,
the diagnostics and historical evidence from the desktop checkout, and the
original ignored scientific inputs and saved results. It also ports the small
curve-fit comparison fix: ground-truth moments are shifted using the true GT
tip position, while estimated moments use the estimated tip position.

`sync_20261008_manifest.json` lists the saved scientific artifacts with their
SHA256 hashes, the three original source snapshots, and files held only in the
backup. Historical result fields and their provenance are preserved; their
manuscript discrepancies are explained in `README.md` and
`recovered_manuscript_crosscheck.md`.

The data/result directories remain ignored for newly generated files. Existing
historical files were explicitly added to Git; Git will track changes to those
files normally. Run diagnostics in a fresh output directory. Deliberately add
new scientific outputs when they are ready to preserve, rather than staging
an environment, a manuscript extraction, or an entire debugging directory.

Before synchronization, both repositories' Git refs were saved in verified
bundles and all working source/data/results were saved in verified ZIPs.
Environments and caches can be recreated and are excluded. The persistent
backup is alongside the Documents/GitHub repo, in
`repo_sync_backups/20261008`. Keep the old desktop repo as an archive until
the second device has adopted this branch and all needed files are confirmed.

## Daily work on either device

Start from a clean working tree on this branch:

```powershell
git status
git pull --ff-only
```

After making and checking your changes, stage the intended files explicitly:

```powershell
git add <files>
git commit -m "Describe the change"
git push
```

Push before changing devices. On the other device, pull before editing. If
`git status` shows work already in progress, preserve or commit that work
before pulling. If `--ff-only` refuses because both devices have new commits,
stop and reconcile those commits; do not use a force push or discard either
working tree.

## Adopt the branch on the second device

First inspect and preserve that device's uncommitted and unpushed work. Then:

```powershell
git fetch origin
git switch --track origin/diagnostic/oracle-ekf-gap
```

If the local branch already exists, switch to it and inspect its relationship
to the remote before pulling. This migration only inspects files on this
computer and GitHub; unique unpushed work on the other device still needs its
own check.

## Diagnostic migration

No full simulation rebuild is required. The original inputs and saved outputs
are preserved, and both recovered diagnostic runners now infer their repo
root and accept a fresh output directory. See `README.md` for commands.
At migration time, the saved EKF derivative patch remained an in-memory
diagnostic comparison; that synchronization did not apply it to production.
The subsequent authorized Step 1 production change and its fresh same-case
validation are documented separately in
[step1_frechet_20261008/README.md](step1_frechet_20261008/README.md).
Historical evidence remains unchanged; consult that report for validation status.

After the production fix, `check_jacobian.py` requires the explicit pre-fix
baseline `--baseline-ref 5312a0fa9c2c5e98399de8a6251fe28df1f616e8`.
It reads that source from Git into memory and compares it with actual
production code; it never restores the old expression to the working tree.
The source comparison permits only the intended two-line Fréchet change
after line-ending normalization. Use a fresh output directory as shown in
`README.md`, and preserve new validation outputs separately from the old runs.

The copied Python environment originally pointed to an old username on the
other device. Its local interpreter and activation paths are repaired using
the existing Python 3.10 installation, retaining the installed packages.
Virtual environments are not stored in Git. On another machine, recreate
one with the pinned `requirements.txt` and use its Python interpreter for
the diagnostic commands.
