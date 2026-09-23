# リモートの GPU 計算機を AiiDA から使う（core.ssh_async + slurm）

AiiDA を動かしている計算機（以下「ホスト」）から、GPU を持つ別の計算機（以下「GPU ノード」）に
MatterSim / SevenNet-Polar のジョブを投げるための手順と、実際に踏んだ落とし穴。root 権限は不要。

## GPU ノード側の準備

1. **conda 環境 `alamode`**（miniforge）。ALAMODE のビルド依存も conda から入れる。
   ```
   conda create -n alamode python=3.11 numpy scipy matplotlib pandas ase spglib phonopy h5py \
       cmake boost-cpp eigen fftw openblas openmpi mkl mkl-devel mkl-include
   ```
   `intel-openmp` は conda-forge に無いので、MKL のスレッド層は `mkl_gnu_thread` + libgomp を使う。
2. **ALAMODE 1.5.0 のビルド**（システムの gcc + conda のライブラリ、rpath を env の lib に向ける）。
   ```
   P=$HOME/miniforge3/envs/alamode; export PATH=$P/bin:$PATH; export MKLROOT=$P
   MKL="$P/lib/libmkl_intel_lp64.so;$P/lib/libmkl_gnu_thread.so;$P/lib/libmkl_core.so;gomp;pthread;m;dl"
   cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_PREFIX_PATH=$P -DSPGLIB_ROOT=$P -DUSE_MKL_FFT=yes -DMPI_HOME=$P \
     -DBLAS_LIBRARIES="$MKL" -DLAPACK_LIBRARIES="$MKL" \
     -DCMAKE_CXX_FLAGS="-I$P/include/eigen3 -I$P/include" \
     -DCMAKE_INSTALL_RPATH=$P/lib -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$P/lib"
   ```
   - cmake 4 は `cmake_minimum_required(3.1)` を拒否する → `CMAKE_POLICY_VERSION_MINIMUM`。
   - 新しい Eigen3Config は `EIGEN3_INCLUDE_DIR` を出さない → `-I$P/include/eigen3` を直接渡す。
   - CMake の FindBLAS は conda 配置の MKL を見つけられない → ライブラリを明示する。
   - anphon は `find_package(MPI REQUIRED)` なので OpenMPI が要る（実行は OpenMP のみでよい）。
   - 同梱の example の入力（`STRUCTURE = POSCAR`）は新しい版の書式で 1.5.0 では動かない。
   OpenBLAS 版（`-DFFTW3_ROOT=$P -DBLA_VENDOR=OpenBLAS`）でも結果は同一だった。
   バイナリと tools/ を `~/bin/alamode/` に置く（ホストと同じ配置にするとコードの登録が楽）。
3. **Python 側**：`pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130`（ドライバに合う CUDA 版）、
   `pip install mattersim`、SevenNet-Polar はソースを送って `pip install ./SevenNet-Polar`（git が無い場合）。
   checkpoint は `~/models/sevennet-polar/` に置く（`SEVENNET_POLAR_MODEL` で変更可）。
4. **aiida-alamode の runner**：GPU ノードには AiiDA は不要。`pip install --no-deps <aiida-alamode のソース>` で
   console script `alamode-ase-runner`（別名 `alamode-mattersim`）だけを使う（`aiida_alamode.ase_runner` は aiida を import しない）。
5. `~/aiida_run` を作る。slurm の GPU は `gres.conf` に登録されていること（`sinfo -o "%G"` で `gpu:1` が見える）。

## ホスト側の登録

```
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519_aiida      # daemon 用（agent に頼らない）
# 公開鍵を GPU ノードの ~/.ssh/authorized_keys に追加
```
`~/.ssh/config`（後述の落とし穴の対策込み）:
```
Host gpu-node
    HostName <IP>
    User <user>
    IdentityFile ~/.ssh/id_ed25519_aiida
    IdentitiesOnly yes
    ServerAliveInterval 60
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 30m
    SetEnv LC_ALL=C
```
```
verdi computer setup -n --label gpu-node-async --hostname <IP> --transport core.ssh_async --scheduler core.slurm \
    --work-dir /home/<user>/aiida_run/ --mpirun-command "/home/<user>/miniforge3/envs/alamode/bin/mpirun -np {tot_num_mpiprocs}" \
    --mpiprocs-per-machine 8 --shebang "#!/bin/bash"
verdi computer configure core.ssh_async gpu-node-async -n --host gpu-node --backend openssh --max-io-allowed 4 --safe-interval 2 --use-login-shell
verdi computer test gpu-node-async
PRE="export PATH=/home/<user>/miniforge3/envs/alamode/bin:\$PATH"
for spec in "alm:alamode.alm_suggest:/home/<user>/bin/alamode/alm" "anphon:alamode.anphon:/home/<user>/bin/alamode/anphon" \
            "displace:alamode.displace_pf:/home/<user>/bin/alamode/displace.py" \
            "analyze_phonons:alamode.analyze_phonons:/home/<user>/bin/alamode/analyze_phonons" \
            "mattersim:alamode.forces_ase:/home/<user>/miniforge3/envs/alamode/bin/alamode-mattersim"; do
  IFS=: read label plugin exe <<< "$spec"
  verdi code create core.code.installed -n --label $label --computer gpu-node-async --default-calc-job-plugin $plugin \
      --filepath-executable $exe --prepend-text "$PRE"
done
```

## ドライバの使い方

```
python run_alamode_phonons.py --preset Si --computer gpu-node-async --gpu --njobs 1 --cores 4
```
- `--computer` はコードのラベル `alm@<computer>` などに使う（既定は環境変数 `AIIDA_ALAMODE_COMPUTER`、無ければ localhost）。
- `--gpu` は MatterSim / SevenNet のジョブに `#SBATCH --gres=gpu:1`（`metadata.options.custom_scheduler_commands`）を付ける。
- GPU が 1 枚なら `--njobs 1`（力計算を分割しても直列になる）。`--cores` は slurm の `MaxCPUsPerNode` 以下にする。

## 落とし穴

| 症状 | 原因 | 対策 |
|---|---|---|
| アップロードが 5 回失敗して一時停止（`whoami` exit 255） | core.ssh_async が同時に多数の ssh 接続を開き、sshd の `MaxStartups`（既定 10）で切られる | `ControlMaster auto` で接続を多重化、`--safe-interval 2 --max-io-allowed 4`。停止したものは `verdi process play <pk>` |
| 再試行で `Failed to create directory ... lost+found` | GPU ノードのロケールが日本語だと `mkdir` が「ファイルが存在します」と返し、AiiDA は英語の "File exists" しか既存扱いしない | `SetEnv LC_ALL=C`（sshd は既定で `LC_*` を受け付ける） |
| ジョブが `InvalidAccount` で数分待つ | slurm の会計設定の癖。害はない | 待つ |
| daemon 再起動後に "Transport task upload was cancelled" | 再起動時に転送タスクが切られた | 自動で再開される。`verdi process play` でも可 |
| 1 秒未満のジョブが GPU で遅い | CUDA の起動に 0.5 秒 | 小さな調和計算はホストの CPU、MD と大きな supercell は GPU |
| anphon が `std::length_error` で落ちる | SCPH が最低温度で発散（"negative frequency is detected" が続く） | ビルドの問題ではない。TMIN を上げる、ADD_HESS_DIAG や MIXBETA_COORD で安定化 |

## 実行時間の記録

- プロセスノードの ctime と mtime：投入から終了までの実時間（キュー待ちとポーリングを含む）。
- `results["timing"]`（alm、anphon、v0.10）：標準出力の開始と終了時刻、`elapsed_seconds`、OpenMP と MPI の数。
- MatterSim 系の `results`：`time_model_load` と `time_total`。
- `detailed_job_info`（sacct）は slurm の会計が無効だと空。

## 今日の例の CPU と GPU の比較（2026-09-23）

| 例 | 指標 | ホスト（CPU 4 スレッド） | GPU ノード（RTX 3060） |
|---|---|---|---|
| Si | κ(300 K) [W/mK] | 149.5 | 148.7 |
| PbTe | NA3 の最高振動数 [THz] | 3.4073 | 3.4075 |
| BaHfO₃ | κ(300 K) [W/mK] / 虚数モード | 8.33 / なし | 8.32 / なし |
| BaZrO₃ | R 点の回転モード [THz] | −1.378 | −1.378 |
| ZrO₂ | NA3 の最高振動数 [THz] | 23.764 | 23.763 |
| BaTiO₃ SCPH | 極性相が消える温度 | 350〜400 K | 350〜400 K |
| ZnO QHA | u_xx / u_zz (1000 K) | 0.00863 / 0.00946 | 0.00863 / 0.00946 |

GPU の力の差は 10⁻⁵ eV/Å 程度で、調和量への影響は 0.1 % 以下、κ で 0.5 %。MD を経る非調和 IFC では
CV の最適 α が変わる（BaTiO₃ 2.5×10⁻⁶ → 4.0×10⁻⁶）が結論は同じ。

| 工程 | CPU | GPU |
|---|---|---|
| MD 5000 step（40 原子）+ 80 構造 | 240〜330 s | 66 s |
| 力計算 9 構造（96 原子） | 数十 s | 1 s |
| 1〜10 構造の力計算、体積緩和 | 0.1〜0.7 s | 0.5〜0.9 s |
| ジョブごとの slurm と ssh のオーバーヘッド | 数 s | 30〜60 s |
