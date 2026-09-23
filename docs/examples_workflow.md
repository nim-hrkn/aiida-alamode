# example のドライバが何をしているか

`example/run_alamode_phonons.py`、`run_alamode_scph.py`、`run_alamode_qha.py` は ALAMODE の tutorial の手順を、
DFT の代わりに MatterSim（または他の ASE calculator）で、全部 AiiDA のプロセスとして実行する。
各物質の物理的な意味は [examples_materials.md](examples_materials.md) を参照。

## 共通の仕組み

- **実行の単位**。alm、anphon、displace.py、analyze_phonons、alamode-mattersim の各実行は AiiDA の CalcJob として
  slurm に投げる。構造の変換や図の作成は calcfunction。全部が provenance graph に残り、最後に
  `verdi node graph generate <pk>` で辿れる。
- **再実行の省略**。各ステップの結果ノードの pk を `<root>/<name>/.node.json` に記録し、次回は完了済みのステップを
  読み飛ばす。途中で失敗しても直したところから続けられる。`--force` で全部やり直す。
- **calculator の固定**。`--calculator` と `--calculator-kwargs` を Dict ノードにして保存し、同じ run ディレクトリで
  別の calculator を指定するとエラーにする。
- **構造の前処理**。入力構造は MatterSim で緩和し、spglib で基本胞を取り出し、`aiida_alamode.io.supercell.make_diagonal_supercell`
  で supercell を作る。ASE の `make_supercell` ではなく、各原子を x_prim + T（T = 0 が先頭）の順で並べる。anphon は
  「translation 1 が基本胞の位置」と仮定するため、Ewald 補正（NONANALYTIC = 3）で必要になる。
- **緩和後の理想化**。緩和した構造には 10⁻⁶ 程度のノイズが残り、alm が空間群を見つけても「自由な IFC が 0 個」と判定する。
  `idealize_structure` が spglib で対称化し、入力の座標系に回転を戻す（SCPH と QHA のドライバで使用）。
- **調和 IFC の共通ステップ**。
  1. `alamode.alm_suggest`：supercell から変位パターンを求める（List として出力）。
  2. `alamode.displace_pf`：supercell を QE 形式の雛形に書き、displace.py -pf で変位構造を作り、TrajectoryData として返す。
  3. `alamode.force_simulator_mattersim`：変位構造を njobs 個の slurm ジョブに分けて MatterSim で力を計算し、結果を結合して
     DFSET（Ry と Bohr の単位、extract.py --QE と同じ書式）を List で返す。
  4. `alamode.alm_opt`：DFSET から IFC を最小二乗で決め、anphon 用の xml を返す。

## 1. run_alamode_phonons.py：調和フォノンと熱伝導率（tutorial 3、5〜7）

任意の構造ファイルと supercell の大きさを受け取る。`--preset Si` と `--preset PbTe` は tutorial の設定。

調和部分
1. `alamode.mattersim_relax` で体積を緩和する（`--relax full` で形状と原子位置も）。Si では a = 5.464 Å になる。
2. 基本胞と supercell を作る。Si は慣用胞の 2×2×2（64 原子）、PbTe は fcc 基本胞の 4×4×4（128 原子）。
3. 共通ステップで調和 IFC を求める。
4. `alamode.anphon` の phonons モードでバンド（Γ-X-Γ-L の tutorial の経路）と DOS（20×20×20）を計算する。`--nonanalytic` に
   複数の値を渡すと NONANALYTIC ごとに計算する。PbTe では 0〜3 の 4 通りで、BORNINFO（誘電率と Born 有効電荷）を渡す。
5. `--ref-xml` があれば、tutorial の DFT の IFC xml も同じ anphon ステップにかける。xml の supercell から基本胞を取り出す
   `primitive_from_fcsxml` を使い、座標系を xml に合わせる。
6. 図 `<name>_phband_phdos.png` と、最大振動数や Γ 点の振動数の要約を出す。

立方（cubic）部分（`--cubic`、Si のプリセットで有効）
7. NORDER = 2、cutoff 7.5 Bohr で alm suggest をやり直し、`select_patterns` で 3 次のパターンだけを選ぶ。変位 0.04 Å で displace する。
8. 力を計算し、`alamode.alm_opt` で調和 IFC を固定（FC2XML）して 3 次 IFC を決める。
9. `alamode.anphon` の RTA モードで κ(T) と κ のスペクトル（KAPPA_SPEC = 1）を 10×10×10 で計算する。
10. `alamode.analyze_phonons` で 300 K のフォノン寿命、平均自由行程に対する累積 κ、境界散乱（1 mm）付きの κ を出す。
11. 図 `<name>_kappa.png` を作る。DFT の 3 次 IFC（`--ref-cubic-xml`）があれば同じ計算を重ねる。

## 2. run_alamode_scph.py：BaTiO₃ の SCPH 構造緩和（tutorial 7.1 と 7.4）

1. 体積緩和、理想化、2×2×2 supercell（40 原子）。
2. 共通ステップで調和 IFC を求める（FC2XML になる）。
3. `alamode.mattersim_md` で 300 K の NVT（Langevin）MD を 1 fs × 5000 step 走らせる。tutorial の
   `displace.py -md -e 1001:5000:50 --random --mag 0.04` と同じく、50 step ごとの 80 個のスナップショットに 0.04 Å の
   ランダム変位を足し、TrajectoryData として返す。
4. その 80 構造の力を計算して DFSET を作る。
5. `alamode.alm_cv`：NORDER = 3、NBODY 2 3 3、cutoff は 3 次 15 Bohr と 4 次 9 Bohr、elastic-net（LASSO）で 4 分割の
   cross validation を α = 10⁻⁸〜10⁻² の 30 点で行い、CV スコアが最小の α を返す。
6. `alamode.alm_opt` にその α を渡して非調和 IFC を決める（FCSXML）。
7. `alamode.anphon` の SCPH モードで、RELAX_STR = 1 の構造緩和付き自己無撞着フォノン計算を 50〜400 K、25 K 刻みで行う。
   `param` の &scph、&relax、&displace の各節はそのまま入力に書かれる。&displace には tutorial の初期変位（Ti を +z、O を −z）を与える。
8. 出力ファイル一式を FolderData で受け取り、原子変位と自由エネルギーの温度依存の図 `<name>_scph_relax.png` を作る。
   Ti の z 変位が消える温度を転移温度の目安として出す。

## 3. run_alamode_qha.py：ZnO の QHA 熱膨張（tutorial 7.5）

1. 完全緩和（格子と原子位置）と理想化。
2. **大きな supercell の調和 IFC**：4×4×2（128 原子）で共通ステップを行い、FC2XML にする。
3. **小さな supercell の非調和 IFC**：3×3×2（72 原子）で、まず調和 IFC を求め、次に 500 K の MD から 80 構造を作り、
   CV と最適化で 3 次と 4 次の IFC を決める（cutoff は 12 と 8 Bohr）。これが FCSXML。
4. **ひずみ下の調和 IFC**：基本胞に xx、yy、zz は 0.005、yz、zx、xy は 0.0025 の 6 種類のひずみを `strain_structure` で与え、
   それぞれ 4×4×2 の supercell で調和 IFC を求める。ひずんだ格子は平衡でないため、`subtract_offset` で無変位構造の力を差し引く。
5. **弾性定数**：`alamode.mattersim_elastic` で基本胞のエネルギーを有限差分し、clamped-ion の 2 次（SOEC、81 成分）と
   3 次（TOEC、729 成分）の弾性定数と、ひずみと力の結合（strain_force.in）を ALAMODE の書式で出す。
6. `make_strain_ifc_folder` で elastic_constants.in、strain_force.in、strain_harmonic.in と 6 個の xml を一つの FolderData に
   まとめ、anphon の STRAIN_IFC_DIR として渡す。
7. `alamode.anphon` の QHA モードで、RELAX_STR = 2 の構造最適化を QHA_SCHEME 0（full）、1（ZSISA）、2（v-ZSISA）の 3 通り、
   0〜1000 K で行う。
8. 各スキームの umn_tensor（熱ひずみ u_xx と u_zz）を図 `<name>_thermal_strain.png` にし、tutorial の DFT の値を破線で重ねる。

## v0.10 で変わった点

- 変位構造と DFSET がファイルではなく TrajectoryData と List で DB に入るため、cwd を指定しなくても provenance だけで
  再現できる。cwd を渡すと結果ファイルも run ディレクトリに置かれる。
- extract.py は使わず、MatterSim の結果から直接 DFSET を作る。
- 上流の設計に合わせて、alm は suggest、opt、cv の 3 クラス、displace は pf クラスを使う。
