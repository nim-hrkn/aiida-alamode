# MCP サーバ（`alamode-mcp`）：LLM から aiida-alamode を使う

`aiida_alamode/mcp_server.py` は [Model Context Protocol](https://modelcontextprotocol.io) のサーバで、
Claude Code / Claude Desktop などの LLM エージェントから、フォノン計算の投入・監視・結果の取得と
AiiDA の provenance の参照をツールとして行える。リポジトリ直下の `.mcp.json` に登録してあるので、
Claude Code をこのリポジトリで起動すると自動で使える。

## インストールと起動

```
pip install -e .[mcp]          # mcp >= 2（Python SDK）
alamode-mcp                    # stdio（Claude Code / Desktop が起動する。手で打つ必要はない）
alamode-mcp --transport streamable-http     # HTTP（127.0.0.1:8000）で常駐させたいとき
```

`.mcp.json`（リポジトリ直下、Claude Code が読む）：

```json
{"mcpServers": {"aiida-alamode": {"command": "alamode-mcp", "args": [], "env": {}}}}
```

`alamode-mcp` は AiiDA の入った Python（conda の `alamode` 環境）の PATH にあること。別の環境から
起動するなら `command` を `/path/to/envs/alamode/bin/alamode-mcp` にする。

`.mcp.json` はリポジトリ直下で Claude Code を起動したときだけ読まれる（親ディレクトリや別のプロジェクトからは
読まれない）。どこから起動しても使えるようにするには user スコープに登録する（`~/.claude.json` に入る、ホスト固有なので
リポジトリには入れない）：

```
claude mcp add --scope user --transport stdio aiida-alamode -- /path/to/envs/alamode/bin/alamode-mcp
claude mcp get aiida-alamode        # Status: Connected を確認
```

登録後は Claude Code を起動し直す（起動中のセッションは `/mcp` で再接続）。環境変数：

| 変数 | 意味 | 既定 |
|---|---|---|
| `AIIDA_PROFILE` | AiiDA プロファイル | 既定プロファイル |
| `AIIDA_ALAMODE_COMPUTER` | 計算機ラベル（`--computer` の既定） | なし（ツールの引数で渡す） |
| `AIIDA_ALAMODE_EXAMPLE_DIR` | driver のある場所 | `<repo>/example` |
| `AIIDA_ALAMODE_RUN_ROOT` | 結果とログを置く root | `<example>/run_mcp` |

## ツール

| ツール | すること |
|---|---|
| `check_packages(computer, names)` | `ase_runner@<computer>` の計算機にある任意パッケージ（mattersim, sevennet-polar, equivar, anisonet, …）。`alamode-ase-runner --check` を transport 越しに実行 |
| `list_codes()` | 計算機ごとの ALAMODE の code（alm, anphon, displace, analyze_phonons, ase_runner）と揃っているか |
| `run_phonons(structure, supercell, …)` | `run_alamode_phonons.py` をバックグラウンドで起動。緩和 → alm suggest → displace → 力 → alm opt → anphon band/DOS（→ cubic IFC + RTA κ）。Z* と ε∞ は `borninfo_calculator` / `dielectric_model` / `dielectric` / `born_charges` / `borninfo` で指定。図の形式は `figure_format=["svg"]`（既定 png、pdf も可） |
| `run_driver(driver, args)` | phonons / scph / qha の driver を生の引数で起動 |
| `run_status(run_id)` | driver の生死、ログ末尾、`.node.json` に記録された AiiDA ノードの状態（waiting / finished / failed と exit code） |
| `run_results(run_dir)` | 緩和後の格子、BORNINFO に入った Z* と ε∞（出所つき）、alm の fitting error、NONANALYTIC ごとの Γ 点振動数と LO-TO シフト、虚数モードの有無、熱力学（ZPE, C_v/3Nk_B, S）、κ、図。値ごとに読み出した AiiDA ノードの pk |
| `run_report(target)` | HTML レポート（`alamode-report`）。target は構造（任意のノード）の pk か run ディレクトリ。provenance を下って何をしたかを自動検出し、full formula・空間群・原子数・Wyckoff 位置、緩和、fit、Z* と ε∞、Γ 点振動数、熱力学、κ、SCPH / QHA、SVG の図（inline）、プロセス表とプロセスグラフを 1 ファイルに書く。返すのは HTML のパスと要点（`report.summary`：値ごとに pk）で、集めた全データは HTML の隣の `<stem>.json` に書いてそのパス（`full_result`）を返す |
| `list_runs(root)` | root 下の run（図ファイル .png / .svg / .pdf の一覧つき）と、このサーバが起動した driver |
| `process_info(pk)` | AiiDA プロセス 1 件：状態、exit code とメッセージ、入出力リンク、report、リモートの作業ディレクトリ |
| `kill_run(run_id)` | driver を止める（投入済みの AiiDA プロセスは残る。`verdi process kill <pk>`） |

リソース：`aiida-alamode://skill`（`.claude/skills/aiida-alamode/SKILL.md`：仕組み、落とし穴、参考値）、
`aiida-alamode://docs/{name}`（`docs/<name>.md`）。

## 典型的な流れ（エージェントが行う）

1. `check_packages(computer="host")` で MatterSim と Z* / ε∞ のモデルの有無を見る。
2. `run_phonons(structure="BaZrO3_Pm-3m.cif", supercell=[2,2,2], name="BaZrO3", nonanalytic=[0,3],
   borninfo_calculator="sevennet-polar", dielectric_model="anisonet", computer="host")`。
   学習元素の外なら `born_charges={"Mg": [1.96], "O": [-1.96]}` と `dielectric=[3.0]` か `dielectric_model="anisonet"`。
3. `run_status(run_id)` を終わるまで繰り返す（`done` が true、`alive` が false）。失敗なら `nodes` の
   `failed` / exit code を見て `process_info(pk)` で report を読む。
4. `run_results(run_dir)` で Γ 点の LO シフト、虚数モード、Z*、ε∞、熱力学を取り、pk とともに報告する。

driver は完了したステップを `.node.json` から再利用するので、同じ `name` で条件を足して再実行しても
最初からやり直しにはならない。

## 設計上の注意

- サーバは driver をサブプロセスで起動するだけで、AiiDA のプロセスは daemon が走らせる。サーバを止めても
  計算は続き、`run_status` は起動記録（`<run root>/.mcp_runs.json`）とログから状態を復元する。
- ツールは AiiDA プロファイルを最初の呼び出し時に読み込む（`alamode-mcp --help` はプロファイル無しで動く）。
- 返り値の大きさ：JSON が `AIIDA_ALAMODE_MCP_MAX_CHARS`（既定 20000 文字）を超える結果はファイルに書き、
  要約（`run_report` は `report.summary`、他はトップレベルのキーと各キーの文字数）と `full_result`（そのパス）だけを返す
  （`run_results` → `<run>/<name>_results.json`、`process_info` → `<run root>/process_<pk>.json`、`list_runs` →
  `<run root>/list_runs.json`）。`run_report` は大きさに関係なく常に全データを `<stem>.json` に書く。エージェントは
  必要な所だけ grep / 部分読みする。ZnO の QHA run の全データは約 30 万文字あり、そのまま返すと文脈に入らない。
- 結果はログではなく AiiDA ノード（`figure` / `thermo_figure` / `figure_kappa` の `summary`、`borninfo_wc` の `results`、
  anphon の `.bands`）から読む。数値の出所は常に pk で示す。
- 動作確認（2026-09-26）：mcp 2.2.0、stdio で tools / resources の一覧、`check_packages`、`run_phonons`（NaCl、
  キャッシュ再利用）、`run_status`、`run_results`、`list_runs`、`process_info` が通ることを確認。
