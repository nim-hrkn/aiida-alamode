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
