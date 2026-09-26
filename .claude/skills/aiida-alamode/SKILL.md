---
name: aiida-alamode
description: aiida-alamode v0.10 で ALAMODE のフォノン計算（調和、立方 IFC + RTA の κ、SCPH、QHA、SevenNet-Polar / Equivar の Born 電荷、調和近似の C_v・S・F(T)）を MatterSim などの ASE calculator と AiiDA で回す・診るときに使う。entry point と入出力、ドライバの使い方、リモート GPU 計算機（core.ssh_async + slurm）の登録、ALAMODE の conda ビルド、既知の落とし穴（supercell の像、BORNINFO の順序、緩和後の理想化、withmpi、sshd の MaxStartups、日本語ロケール、SCPH の発散）。
---

# aiida-alamode（v0.10）

ALAMODE の alm / anphon / displace.py / analyze_phonons と、MatterSim（または mace, chgnet, sevennet, orb, emt）の
力計算、SevenNet-Polar / Equivar の Born 電荷を、すべて AiiDA の CalcJob / WorkChain として実行するプラグイン。
詳しい説明は `docs/`（`examples_workflow.md`、`examples_materials.md`、`born_effective_charges.md`、`remote_gpu_computer.md`、`rabbitmq.md`、`mcp_server.md`）。

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
| alamode.bec_ase / epsinf_ase | 誘電特性の予測（`calculations/dielectric_calcjob.py`：Z* は BornChargesBaseCalculation、ε∞ は DielectricTensorBaseCalculation の下；calculator `sevennet-polar` / `equivar`、AnisoNet、将来は VASP/QE） |
| alamode.borninfo（BornInfoWorkChain） | Z* のジョブ + ε∞ のジョブ（または `dielectric` の値）→ calcfunction make_borninfo → `borninfo`。名前空間 `bec` / `epsinf` と `bec_plugin` / `epsinf_plugin` でエンジンを選ぶ | structure(s), calculator Dict → arrays / structure / displaced_structures / strain_ifc_folder / borninfo |
| alamode.anphon | anphon | structure(prim), fcsxml, mode(phonons/RTA/SCPH/QHA…), param, borninfo, fc2xml, extra_files → phband_file, phdos_file, kl_file, output_folder, results(timing)。DOS 実行（phonons_mode='dos'）は `thermo`（ArrayData: temperatures, heat_capacity, entropy, internal_energy, free_energy、基本胞あたり、単位は属性 `units`）も出す |
| alamode.analyze_phonons | analyze_phonons | file_result, calc(tau/cumulative/kappa_boundary), param → *_file |

## ドライバ（example/）

- `run_alamode_phonons.py --preset Si|PbTe` / `--structure X.cif --supercell n n n [--relax full --idealize] [--cubic --cubic-cutoff BOHR] [--nonanalytic 0 3 --borninfo FILE | --borninfo-calculator sevennet-polar|equivar (--dielectric-model anisonet | --dielectric E) | --born-charges Mg:1.96 O:-1.96 --dielectric-model anisonet]`
- `run_alamode_scph.py`（BaTiO₃ 既定。MatterSim では TMAX 700、DT 50、MIXBETA_COORD 0.2 が必要）
- `run_alamode_qha.py`（ZnO 既定）、`run_BaHfO3_example.sh`（Z* → フォノン → κ → SCPH の一括）
- MCP サーバ `alamode-mcp`（`aiida_alamode/mcp_server.py`、`.mcp.json` に登録、`pip install -e .[mcp]`）：check_packages / list_codes / run_phonons / run_driver / run_status / run_results / list_runs / process_info / kill_run。driver をサブプロセスで起動し、結果は AiiDA ノードから pk つきで返す（docs/mcp_server.md）。
- 必須は ALAMODE と aiida-core だけ。MatterSim / SevenNet-Polar / Equivar / AnisoNet は example とテストのための任意パッケージ（ase_runner の code が走る計算機に入れる）。
  `example/check_packages.py [--computer X] [--require ...]`（= その計算機で `alamode-ase-runner --check`）で有無を確認。各 driver は投入前に同じ確認をして、無ければメッセージを出して止まる。`run_all_examples.sh` は無い example を飛ばす。
- `run_alamode_phonons.py` の図：`<name>_phband_phdos.png`（NA0 / NA3 のバンドと DOS）、`<name>_kappa.png`、`<name>_thermo.png`（C_v(T) と Dulong–Petit 3Nk_B、S(T)、F(T) 零点込み。NONANALYTIC 最大の DOS 実行の `thermo` から。ログの `thermo figure:` に ZPE、100/300/1000 K の値、C_v が 0.9×3Nk_B に達する温度）。
- 共通：`--computer <label> --gpu --njobs N --cores N --root DIR`。`provenance_processes.py <pk> out.png`（拡張子 .svg / .pdf も可）でプロセスだけの provenance 図。
- レポート：`alamode-report <run dir | 構造の pk> [-o x.html] [--summary | --json]`（`aiida_alamode/report.py`、MCP は `run_report`、driver は終了時に自動で書く）。HTML は要点カード（式、空間群、Wyckoff 位置、Γ 最高モード、κ、ZPE）、セルの表、緩和、fit、Z*/ε∞、フォノン、熱力学、κ、SCPH/QHA、図、プロセス表（折り畳み）とプロセスグラフ。手順の箇条書きとサイト座標の表は載せない（2026-09-26 に削除、ユーザーの希望）。MCP の `run_report` は `summary()` の要点と `<stem>.json`（全データ）のパスを返す。MCP の返り値は 20000 文字を超えるとファイルに書かれ、パスだけ返る（`_deliver`）。`.node.json` のラベルには頼らず、構造の pk から provenance を上（入力ファイル → 入力セル → 緩和 → 基本胞 → supercell）と下（全プロセス）にたどり、entry point / 関数名で種類を判定して（`classify`）、各プロセスの入出力から何をしたかを 1 行ずつ書く（`describe_step`）。数値はすべてノードから読み、図は figure calcfunction の `img_file`（SVG があれば inline、無ければ PNG を base64）。graphviz があればプロセスグラフも入れる。
- **CalcJob / calcfunction / WorkChain を追加・変更したら `report.py` も更新する**：`classify`（entry point か process_label → kind）、`describe_step`（入出力 → 1 行の説明）、`collect` の該当節（新しい出力キーや単位）、必要なら `render` の表。入出力のキー名を変えたときも同じ（`results` / `result`、`phband_file`、`thermo` ArrayData、`kl_file`、`img_file` などを直接読んでいる）。変えたら `alamode-report` を phonons（Si）、LO-TO（MgO）、SCPH、QHA の run で回して、手順の一覧に「(AttributeError: …)」が出ないことと図が入ることを確認する。
- 図の形式：3 つの driver とも `--figure-format svg`（`pdf`、複数指定 `png svg` も可）でベクタ図を書く。終わった run に別形式を指定すると図の calcfunction だけ走る（bank のキーは `figure[svg]` など）。MCP の `run_phonons` は `figure_format=["svg"]`。img workchain（phband_img / phdos_img / freeenergy_img）は `img_filename` の拡張子で決まる。

## 落とし穴（順に疑う）

1. **supercell の像**：`make_diagonal_supercell` を使う（ASE の make_supercell だと translation 1 が基本胞の位置とずれ、NONANALYTIC=3 が壊れる）。
2. **緩和後のノイズ**：alm が「自由な IFC 0 個」→ `--idealize`（spglib で対称化して入力の座標系に戻す）。
3. **BORNINFO の順序**：&position（構造の原子順）であって KD の順ではない。BornInfoWorkChain と anphon に同じ StructureData を渡す。ε∞ は SevenNet-Polar からは出ない。`dielectric_model` = anisonet（電子誘電率を予測、~/models/anisonet/anisonet-stock.ckpt）か `dielectric` 入力で与える。
4. **anphon は NAT を受け付けない**（atoms_to_alm_in で alm モードだけに書く）。ひずんだ格子の IFC は `subtract_offset` が必須。
5. **aiida-core ≥ 2.3 の withmpi**：既定値が無いので base CalcJob で False を設定済み。
6. **SCPH の発散**：TMAX が低い（400 K）と 75 K の構造ループが 1000 回回っても収束しない。最低温度で "negative frequency is detected" が続くと anphon が `std::length_error` で落ちる → TMIN を上げる、ADD_HESS_DIAG。
7. **リモート計算機（core.ssh_async）**：sshd の MaxStartups → `ControlMaster auto`；日本語ロケール → `SetEnv LC_ALL=C`；一時停止は `verdi process play`。
8. **slurm の InvalidAccount**：23.11 の accounting_storage/none の不具合。ジョブは動くがバックフィルだけで 30 秒に 1 本。直し方は docs/slurm_invalidaccount.md。
9. **GPU**：`--gpu`（`#SBATCH --gres=gpu:1`）。1 秒未満のジョブは GPU の方が遅い。MD は 4〜5 倍速い。
10. **SevenNet-Polar の元素**：PS 系は Ba, Ca, Hf, Li, O, P, Pb, Sr, Ti, Zr、PM 系は Li, O, P, Zr。Zn、Si、Te は不可。ASR の残差が大きい（> 0.5 e）なら分布外。Equivar（`--borninfo-calculator equivar`、`aiida_alamode.equivar`、重み `~/models/equivar/BM1.pt`）も同じ学習データなので同じ 10 元素。近接リストは最小像でなく全周期像で作る（5 原子胞で最小像だと Ti の Z* が 4.4 になる）。

## Z* と ε∞（結果の目安、docs/born_effective_charges.md）

- Z*（SevenNet-Polar PS-M）：BaTiO₃ Ba 2.72 / Ti 7.72 / O −2.15, −6.14（DFT 2.75 / 7.16 / −2.11, −5.69）、BaZrO₃ 2.72 / 5.68 / −1.98, −4.44、
  BaHfO₃ 2.74 / 5.42 / −1.99, −4.19、単斜 ZrO₂ Zr 5.0〜5.5 / O −2.5〜−2.8。ASR 残差 > 0.5 e は分布外（ルチル・アナターゼ TiO₂）。
- Z*（Equivar BM1、同じ胞）：BaTiO₃ 2.80 / 7.82 / −2.27, −6.08、BaZrO₃ 2.52 / 5.69 / −1.66, −4.89、BaHfO₃ 2.73 / 5.45 / −2.01, −4.16。
  分布内では SevenNet-Polar と 0.1 e 程度で一致（BaZrO₃ の O∥ は 0.45 e 差）。BM2（小型）は分布外のルチル TiO₂ で崩れる（ASR 残差 3.8 e）。
  重みは TorchScript で torch_scatter の演算子を参照 → `aiida_alamode.equivar.register_scatter_ops()` が素の torch で代替を登録（torch_scatter は不要）。
- 調和近似の熱力学（MatterSim、基本胞あたり）：ZPE Si 121 meV、PbTe 25、BaHfO₃ 249、BaZrO₃ 244、ZrO₂ 796。C_v が 0.9×3Nk_B に達する温度 Si 450 K、PbTe 90 K、
  BaZrO₃ 390 K、ZrO₂ 500 K。Si と PbTe は DFT 参照と 2 % 以内。
- ε∞（AnisoNet、電子誘電率、num_neighbors = 34.956847 に固定）：Si 13.1、MgO 3.13、BaHfO₃ 4.69、BaZrO₃ 4.93、BaTiO₃ 6.3、SrTiO₃ 6.55、
  ZrO₂ 5.2〜5.8、ルチル TiO₂ 7.7 / 9.3。文献の 1〜2 割以内。`example/test_epsinf.py --computer <label>` で再検証。
- NA3 で Γ 点の最高 LO が 4〜6 THz 上がり、TO とソフトモードは動かなければ BORNINFO は正しく入っている。
- 学習元素の外は `--born-charges Mg:1.96 O:-1.96`（文献 Z*）+ `--dielectric-model anisonet` または `--dielectric E`。MgO Γ TO 11.0 / LO 20.2 THz（実験 12.0 / 21.5）、NaCl 4.65 / 7.29（実験 4.9 / 7.9）。
  γ-Li₃PO₄（Pnma 32 原子、`--supercell 2 1 2 --relax full --idealize`）は SevenNet-Polar で Li 1.04 / P 3.0 / O −1.54（DFPT 学習データの平均 1.07 / 2.96 / −1.54）、ε∞ 2.55〜2.58。
- AnisoNet の落とし穴：predict notebook の書き方だと num_neighbors がバッチ依存で値が変わる。runner の固定値を使う。
  mygarden 系の壊れた GPU では `device: auto` が CUDA を選んで失敗するため runner は初期化失敗時に CPU に落ちる。

## 失敗の診かた

`verdi process report <pk>`（例外と exit code）、`verdi calcjob outputcat <pk>`（標準出力）、`verdi node attributes <pk>` の `remote_workdir`。
ドライバのログは `example/run_*.log`（`sys.stdout` は行バッファ）。
