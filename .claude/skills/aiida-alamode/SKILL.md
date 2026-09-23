---
name: aiida-alamode
description: aiida-alamode v0.10 で ALAMODE のフォノン計算（調和、立方 IFC + RTA の κ、SCPH、QHA、SevenNet-Polar の Born 電荷）を MatterSim などの ASE calculator と AiiDA で回す・診るときに使う。entry point と入出力、ドライバの使い方、リモート GPU 計算機（core.ssh_async + slurm）の登録、ALAMODE の conda ビルド、既知の落とし穴（supercell の像、BORNINFO の順序、緩和後の理想化、withmpi、sshd の MaxStartups、日本語ロケール、SCPH の発散）。
---

# aiida-alamode（v0.10）

ALAMODE の alm / anphon / displace.py / analyze_phonons と、MatterSim（または mace, chgnet, sevennet, orb, emt）の
力計算、SevenNet-Polar の Born 電荷を、すべて AiiDA の CalcJob / WorkChain として実行するプラグイン。
詳しい説明は `docs/`（`examples_workflow.md`、`examples_materials.md`、`born_effective_charges.md`、`remote_gpu_computer.md`）。

## まず確認すること

- `verdi status`（daemon と RabbitMQ）、`verdi code list`（alm, anphon, displace, analyze_phonons, ase_runner @<computer>）。
- プラグインを変えたら `pip install -e . --no-deps` と `verdi daemon restart`。計算機側（runner だけ）は `pip install --no-deps <src>`。
- 結果の再利用は `<root>/<name>/.node.json`（ドライバの覚え書き、provenance ではない）。`--force` で全部やり直す。

## entry point

| entry point | 役割 | 主な入力 → 出力 |
|---|---|---|
| alamode.alm_suggest / alm_opt / alm_cv | alm | structure, norder, cutoff, param, dfset(List), fc2xml → pattern(List) / input_ANPHON(xml), results(alpha_min, timing) |
| alamode.displace_pf | displace.py -pf | structure_org, pattern, mag, norder → displaced_structures(TrajectoryData) |
| alamode.forces（ForcesWorkChain、forces_plugin で CalcJob を選ぶ） | 力 + DFSET | code, structures, structure_org, calculator, njobs, subtract_offset → arrays, dfset(List) |
| alamode.forces_ase / relax_ase / md_ase / elastic_ase | 力の予測（`calculations/force_calcjob.py`、基底 ForceCalculatorBaseCalculation；ASE エンジンは `engine_base.py` の AseRunnerBaseCalculation、runner `alamode-ase-runner`。DFT 用は同じ基底の下に足す） |
| alamode.bec_ase / epsinf_ase | 誘電特性の予測（`calculations/dielectric_calcjob.py`：Z* は BornChargesBaseCalculation、ε∞ は DielectricTensorBaseCalculation の下；SevenNet-Polar、AnisoNet、将来は VASP/QE） |
| alamode.borninfo（BornInfoWorkChain） | Z* のジョブ + ε∞ のジョブ（または `dielectric` の値）→ calcfunction make_borninfo → `borninfo`。名前空間 `bec` / `epsinf` と `bec_plugin` / `epsinf_plugin` でエンジンを選ぶ | structure(s), calculator Dict → arrays / structure / displaced_structures / strain_ifc_folder / borninfo |
| alamode.anphon | anphon | structure(prim), fcsxml, mode(phonons/RTA/SCPH/QHA…), param, borninfo, fc2xml, extra_files → phband_file, phdos_file, kl_file, output_folder, results(timing) |
| alamode.analyze_phonons | analyze_phonons | file_result, calc(tau/cumulative/kappa_boundary), param → *_file |

## ドライバ（example/）

- `run_alamode_phonons.py --preset Si|PbTe` / `--structure X.cif --supercell n n n [--relax full --idealize] [--cubic --cubic-cutoff BOHR] [--nonanalytic 0 3 --borninfo FILE | --borninfo-calculator sevennet-polar (--dielectric-model anisonet | --dielectric E)]`
- `run_alamode_scph.py`（BaTiO₃ 既定。MatterSim では TMAX 700、DT 50、MIXBETA_COORD 0.2 が必要）
- `run_alamode_qha.py`（ZnO 既定）、`run_BaHfO3_example.sh`（Z* → フォノン → κ → SCPH の一括）
- 共通：`--computer <label> --gpu --njobs N --cores N --root DIR`。`provenance_processes.py <pk> out.png` でプロセスだけの provenance 図。

## 落とし穴（順に疑う）

1. **supercell の像**：`make_diagonal_supercell` を使う（ASE の make_supercell だと translation 1 が基本胞の位置とずれ、NONANALYTIC=3 が壊れる）。
2. **緩和後のノイズ**：alm が「自由な IFC 0 個」→ `--idealize`（spglib で対称化して入力の座標系に戻す）。
3. **BORNINFO の順序**：&position（構造の原子順）であって KD の順ではない。BornInfoWorkChain と anphon に同じ StructureData を渡す。ε∞ は SevenNet-Polar からは出ない。`dielectric_model` = anisonet（電子誘電率を予測、~/models/anisonet/anisonet-stock.ckpt）か `dielectric` 入力で与える。
4. **anphon は NAT を受け付けない**（atoms_to_alm_in で alm モードだけに書く）。ひずんだ格子の IFC は `subtract_offset` が必須。
5. **aiida-core ≥ 2.3 の withmpi**：既定値が無いので base CalcJob で False を設定済み。
6. **SCPH の発散**：TMAX が低い（400 K）と 75 K の構造ループが 1000 回回っても収束しない。最低温度で "negative frequency is detected" が続くと anphon が `std::length_error` で落ちる → TMIN を上げる、ADD_HESS_DIAG。
7. **リモート計算機（core.ssh_async）**：sshd の MaxStartups → `ControlMaster auto`；日本語ロケール → `SetEnv LC_ALL=C`；一時停止は `verdi process play`。
8. **GPU**：`--gpu`（`#SBATCH --gres=gpu:1`）。1 秒未満のジョブは GPU の方が遅い。MD は 4〜5 倍速い。
9. **SevenNet-Polar の元素**：PS 系は Ba, Ca, Hf, Li, O, P, Pb, Sr, Ti, Zr、PM 系は Li, O, P, Zr。Zn、Si、Te は不可。ASR の残差が大きい（> 0.5 e）なら分布外。

## 失敗の診かた

`verdi process report <pk>`（例外と exit code）、`verdi calcjob outputcat <pk>`（標準出力）、`verdi node attributes <pk>` の `remote_workdir`。
ドライバのログは `example/run_*.log`（`sys.stdout` は行バッファ）。
