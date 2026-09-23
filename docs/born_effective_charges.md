# Born 有効電荷（BEC）を機械学習モデルで求めて anphon に渡す

anphon の LO-TO 分裂の補正（NONANALYTIC = 1, 2, 3）には BORNINFO ファイル（高周波誘電率 ε∞ と各原子の
Born 有効電荷 Z*）が必要になる。MatterSim などの力場は Z* を出せないため、v0.10 では Z* を予測する
モデル SevenNet-Polar を calculator として使い、BORNINFO を作る CalcJob `alamode.mattersim_bec` を追加した。
候補モデルの比較は `alamode_test/BCE.md` を参照。

## SevenNet-Polar のインストール

- コード: https://github.com/AugustinLu/SevenNet-Polar （SevenNet 0.12.2 のフォーク。パッケージ名は `sevenn`
  のままなので、元の SevenNet と同じ環境には入れられない）
- 学習済みモデルとデータ: Zenodo https://doi.org/10.5281/zenodo.21322761

```
pip install git+https://github.com/AugustinLu/SevenNet-Polar.git      # alamode 環境（torch 2.14, e3nn 0.6）で MatterSim と共存できた
mkdir -p ~/models/sevennet-polar && cd ~/models/sevennet-polar
for f in SevenNet-PS-S.pth SevenNet-PS-M.pth SevenNet-PS-L.pth; do
  curl -sL "https://zenodo.org/api/records/21322761/files/$f/content" -o $f
done
```

| checkpoint | 予測量 | 元素 |
|---|---|---|
| SevenNet-PS-S / M / L | Z* のみ（BEC 専用） | Ba, Ca, Hf, Li, O, P, Pb, Sr, Ti, Zr |
| SevenNet-PM-S / M / L | エネルギー、力、応力、Z*（マルチタスク） | Li, O, P, Zr |

どの checkpoint も ε∞ は返さない（calculator 自体は `dielectric_tensor` に対応している）。
ペロブスカイトの学習データは ABO₃（A = Ba, Ca, Sr, Pb、B = Ti, Zr, Hf）の置換系 1,224 構造で、
PbTe のような Te を含む系は扱えない。

`aiida_alamode.mattersim_runner.CALCULATORS` の `sevennet-polar` は既定で
`~/models/sevennet-polar/SevenNet-PS-M.pth`（環境変数 `SEVENNET_POLAR_MODEL` で変更）を読む。

## CalcJob `alamode.mattersim_bec`（`MattersimBecCalculation`）

- 入力: `structure`（基本胞。anphon の &position と同じ原子順）、`calculator`（例 `{"name": "sevennet-polar"}`）、
  `dielectric`（モデルが ε∞ を返さないときの 3×3、対角 3 成分、または等方 1 成分）、`enforce_asr`（既定 True、
  Σ_i Z*_i = 0 になるよう平均を引く）。
- 出力: `results`（Z* の対角、ASR の残差、ε∞ の出所）、`born_effective_charges`（ArrayData: `bec`、`bec_raw`、`dielectric`）、
  `borninfo`（ε∞ が分かるときだけ。anphon の `borninfo` 入力にそのまま渡せる SinglefileData）。
- 実体は runner の `bec` モード（`alamode-mattersim job.json`）で、ASE calculator の
  `results["born_effective_charges"]`（nat×3×3）を読む。

## ドライバでの使い方

```
python run_alamode_phonons.py --structure BaTiO3_Pm-3m.cif --supercell 2 2 2 --name BaTiO3_bec \
    --nonanalytic 0 3 --borninfo-calculator sevennet-polar --dielectric 6.7
```

`--borninfo` でファイルを渡す代わりに `--borninfo-calculator` を指定すると、緩和後の基本胞に対して
`alamode.mattersim_bec` が走り、その BORNINFO が anphon に渡る。`--dielectric` の値は文献値を与える
（立方 BaTiO₃ の ε∞ = 6.7 は Zhong, King-Smith, Vanderbilt, PRL 72, 3618 (1994) の LDA 値）。

## 注意

- **原子の順序**。BORNINFO の Z* は anphon 入力の &position の順（構造の原子順）で、&general の KD の順（Z の昇順）ではない。
  この CalcJob には anphon に渡すのと同じ StructureData を渡すこと。
- **転置の規約**。Z*_{αβ} の添字順は VASP の `BORN EFFECTIVE CHARGES` の行の並びをそのまま書いている
  （学習データが VASP 由来）。立方晶では対称なので影響しないが、低対称の系では確認が要る。
- **ε∞ は別に用意する**。SevenNet-Polar の公開 checkpoint は ε∞ を返さない。
- **精度の目安**。立方 BaTiO₃（a = 4.0 Å）で PS-M は Ba 2.72、Ti 7.74、O −2.15（⊥）/ −6.15（∥）。
  DFT（LDA）の文献値は Ba 2.75、Ti 7.16、O −2.11 / −5.69。ASR の残差は 0.01 e 程度。

## 例：単斜晶 ZrO₂（バデレアイト、P2₁/c）

SevenNet-Polar の学習データに ZrO₂ が入っているので、Z* を得るのに最も適した系である。2 通りの実行例:

```
# 力は MatterSim、Z* は SevenNet-PS-M
python run_alamode_phonons.py --structure ZrO2_P2_1c.cif --supercell 2 2 2 --relax full --idealize --name ZrO2 \
    --nonanalytic 0 3 --borninfo-calculator sevennet-polar --dielectric 4.9 --emax 900
# 力も Z* も SevenNet-PM-M（マルチタスク checkpoint、Li O P Zr のみ）
python run_alamode_phonons.py --structure ZrO2_P2_1c.cif --supercell 2 2 2 --relax full --idealize --name ZrO2_sevennet_pm \
    --calculator sevennet --calculator-kwargs '{"model": "~/models/sevennet-polar/SevenNet-PM-M.pth"}' --calc-label SevenNet-PM-M \
    --nonanalytic 0 3 --borninfo-calculator sevennet-polar --borninfo-kwargs '{"model": "~/models/sevennet-polar/SevenNet-PM-M.pth"}' \
    --dielectric 4.9 --emax 900
```

- `--relax full --idealize`：単斜晶なので格子の形も緩和し、緩和後の 10⁻⁶ のノイズを spglib で除く（無いと alm が
  「自由な IFC 0 個」になる）。
- 単斜晶のバンド経路には M1 や H1 のような 2 文字のラベルがあり、v0.9 の経路生成はそれを 1 文字ずつ分解して落ちた
  （v0.10 で ASE の `parse_path_string` に置き換え）。
- SevenNet の calculator は結果に nat×3×3 の Z* を残すので、runner が書く extxyz のコピーはエネルギー、力、応力だけにした。
- ε∞ = 4.9 は単斜晶 ZrO₂ の DFT 値（4.7〜5.2、Zhao & Vanderbilt, PRB 65, 075105 (2002)）の等方近似。

結果（2026-09-23）:

| | 格子定数 a, b, c [Å] | Z* 対角（Zr / O） | Γ 点の最高振動数 NA0 → NA3 | 虚数モード |
|---|---|---|---|---|
| 実験（Howard 1988） | 5.151, 5.212, 5.317 | — | — | — |
| MatterSim + PS-M | 5.235, 5.249, 5.440 | 5.54, 5.44, 5.01 / −2.5〜−2.8 | 21.6 → 23.8 THz | NA0 で −0.8 THz（Y-D 間）、NA3 で消える |
| SevenNet-PM-M | 5.181, 5.252, 5.363 | 5.52, 5.52, 4.91 / −2.5〜−2.8 | 21.4 → 23.2 THz | なし |

Z* は DFT の文献値（Zr 約 +5.4〜5.7、O 約 −2.3〜−3.2）と同程度で、ASR の残差は 0.1 e 以下。
SevenNet-PM-M の格子定数は実験に近い。図は `example/run_v010/ZrO2*/…_phband_phdos.png`。

## 例：立方 BaZrO₃ と BaHfO₃（5 原子、SevenNet-Polar の分布内）

```
python run_alamode_phonons.py --structure BaZrO3_Pm-3m.cif --supercell 2 2 2 --name BaZrO3 \
    --nonanalytic 0 3 --borninfo-calculator sevennet-polar --dielectric 4.9 --emax 900
python run_alamode_phonons.py --structure BaHfO3_Pm-3m.cif --supercell 2 2 2 --name BaHfO3 \
    --nonanalytic 0 3 --borninfo-calculator sevennet-polar --dielectric 4.9 --emax 900
```

| | 緩和後 a [Å]（実験） | Z*（A / B / O⊥ / O∥） | 調和近似の虚数モード |
|---|---|---|---|
| BaZrO₃ | 4.254（4.192） | 2.72 / 5.68 / −1.98 / −4.44 | R 点の八面体回転（3 重）−1.38 THz。実験格子定数では −2.12 THz（圧縮で強まる） |
| BaHfO₃ | 4.204（4.171） | 2.74 / 5.42 / −1.99 / −4.19 | なし（R 点の最低 2.20 THz） |

- BaZrO₃ の R 点の虚数は PBE 系の DFT でも見られる既知の結果（実験では零点振動と非調和で立方相が 2 K まで保たれる）。
  Γ、X、M 点は安定で、LO-TO 分裂の検証には支障ない。有限温度の安定化は `run_alamode_scph.py` で確認できる。
- BaHfO₃ は Γ-X-M-Γ-R-X-M-R の全経路で虚数なし。NA3 で Γ 点の最高 LO が 14.5 → 19 THz 程度に上がる。
- Z* は DFT の文献値（BaZrO₃: 2.7 / 6.1 / −2.0 / −4.8）と近く、ASR 残差は小さい。ε∞ = 4.9 は文献値。
