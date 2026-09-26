# -*- coding: utf-8 -*-
"""build docs/intro.html of aiida-alamode: bilingual (ja / en), standalone, inline SVG.

Reads example/run_v013/MgO/MgO_report.html (written by the MgO driver run with --figure-format svg) and
MgO_phband_phdos.svg next to it, draws the provenance graph of the alm opt node with verdi (graphviz),
replaces computer names by `host`, and writes docs/intro.html.  Rebuild after changing the report layout
or the MgO example:  python docs/build_intro.py [--profile NAME] [--alm-opt-pk 14077]
"""
import os, re, html, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "intro.html")
MGO = os.path.join(REPO, "example", "run_v013", "MgO")
ALM_OPT_PK = "14077"
PROFILE = None
args = sys.argv[1:]
while args:
    a = args.pop(0)
    if a == "--alm-opt-pk":
        ALM_OPT_PK = args.pop(0)
    elif a == "--profile":
        PROFILE = args.pop(0)

def blur(t):
    t = t.replace(REPO + "/", "<repo>/").replace(REPO.replace("-", "&#45;") + "/", "&lt;repo&gt;/")
    return re.sub(r"mygarden[a-z0-9-]*", "host", t)

with tempfile.TemporaryDirectory() as tmp:
    cmd = ["verdi"] + (["-p", PROFILE] if PROFILE else []) + ["node", "graph", "generate", ALM_OPT_PK,
           "--ancestor-depth", "1", "--descendant-depth", "1", "-f", "svg", "-O", os.path.join(tmp, "graph")]
    subprocess.run(cmd, check=True, capture_output=True)
    graph = blur(open(os.path.join(tmp, "graph")).read())
graph = graph[graph.index("<svg"):]
graph = re.sub(r'<svg width="[^"]*" height="[^"]*"', '<svg', graph, count=1)
phband = open(os.path.join(MGO, "MgO_phband_phdos.svg")).read()
phband = phband[phband.index("<svg"):]
report = open(os.path.join(MGO, "MgO_report.html")).read()
report_main = blur(re.search(r"<main>(.*)</main>", report, re.S).group(1))

# ------------------------------------------------------------------ pieces
def J(s): return f'<div class="lang ja">{s}</div>'
def E(s): return f'<div class="lang en">{s}</div>'

def arch_svg(lang):
    ja = lang == "ja"
    t = dict(
        claude_sub="LLM（人が日本語で指示）" if ja else "LLM (the person gives instructions)",
        mcp_sub="alamode-mcp", plug_sub="CalcJob / WorkChain / driver / report",
        aiida_sub="来歴・投入・保存" if ja else "provenance, submission, storage",
        comp="計算機" if ja else "computer",
        comp_sub="SLURM（local / ssh）で alm, anphon, runner" if ja else "alm, anphon, runner under SLURM (local / ssh)",
        runner="ASE runner (alamode-ase-runner)",
        runner_sub="MatterSim の力、SevenNet-Polar の Z*、AnisoNet の ε∞" if ja else "forces (MatterSim), Z* (SevenNet-Polar), ε∞ (AnisoNet)",
        alamode="ALAMODE", alamode_sub="alm / displace / anphon / analyze_phonons",
        a1="力・Z*・ε∞ の計算を委ねる" if ja else "delegates forces, Z*, ε∞",
        a2="インストール済みの ALAMODE を実行" if ja else "runs the installed ALAMODE",
        foot=("実線: 呼び出し、破線: 計算機にあらかじめインストールされたプログラムをジョブとして実行する。runner も計算機上のジョブ（code ase_runner@host）。人は左端の Claude にだけ話す。"
              if ja else "solid: calls; dashed: runs programs installed on the computer as jobs. The runner is also a job there (code ase_runner@host). The person talks only to Claude."),
    )
    return f'''<div class="fig" style="max-width:980px"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 310" font-family="-apple-system,'Segoe UI',Roboto,'Noto Sans JP',sans-serif" font-size="15" role="img" aria-label="architecture">
<defs><marker id="arr{lang}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#52514e"/></marker></defs>
<rect width="980" height="310" fill="#ffffff"/>
<g stroke="#2a4a7a" stroke-width="1.5">
<rect x="20" y="60" width="130" height="64" rx="8" fill="#eef5ff"/><rect x="200" y="60" width="110" height="64" rx="8" fill="#f4f4f1"/>
<rect x="360" y="60" width="170" height="64" rx="8" fill="#e6f4ec"/><rect x="580" y="60" width="130" height="64" rx="8" fill="#f4f4f1"/>
<rect x="760" y="60" width="200" height="64" rx="8" fill="#f4f4f1"/>
<rect x="330" y="195" width="230" height="64" rx="8" fill="#e6f4ec"/><rect x="760" y="195" width="200" height="64" rx="8" fill="#fff7e6"/></g>
<g text-anchor="middle" fill="#0b0b0b">
<text x="85" y="86">Claude</text><text x="85" y="108" font-size="11" fill="#52514e">{t['claude_sub']}</text>
<text x="255" y="86">MCP</text><text x="255" y="108" font-size="12" fill="#52514e">{t['mcp_sub']}</text>
<text x="445" y="86" font-weight="bold">aiida-alamode</text><text x="445" y="108" font-size="11" fill="#52514e">{t['plug_sub']}</text>
<text x="645" y="86">AiiDA</text><text x="645" y="108" font-size="11" fill="#52514e">{t['aiida_sub']}</text>
<text x="860" y="86">{t['comp']}</text><text x="860" y="108" font-size="11" fill="#52514e">{t['comp_sub']}</text>
<text x="445" y="221" font-weight="bold" font-size="14">{t['runner']}</text><text x="445" y="243" font-size="11" fill="#52514e">{t['runner_sub']}</text>
<text x="860" y="221" font-weight="bold">{t['alamode']}</text><text x="860" y="243" font-size="11" fill="#52514e">{t['alamode_sub']}</text></g>
<g stroke="#52514e" stroke-width="1.6" fill="none" marker-end="url(#arr{lang})" marker-start="url(#arr{lang})">
<line x1="150" y1="92" x2="200" y2="92"/><line x1="310" y1="92" x2="360" y2="92"/><line x1="530" y1="92" x2="580" y2="92"/><line x1="710" y1="92" x2="760" y2="92"/></g>
<g stroke="#52514e" stroke-width="1.6" fill="none" marker-end="url(#arr{lang})">
<line x1="445" y1="124" x2="445" y2="195"/><line x1="860" y1="124" x2="860" y2="195" stroke-dasharray="5 4"/><line x1="790" y1="124" x2="560" y2="195" stroke-dasharray="5 4"/></g>
<g font-size="12" fill="#52514e"><text x="455" y="165">{t['a1']}</text><text x="870" y="165">{t['a2']}</text>
<text x="20" y="295" font-size="11">{t['foot']}</text></g></svg></div>'''

# ------------------------------------------------------------------ page
head = '''<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>aiida-alamode 入門 / Introduction</title>
<style>
body{font-family:-apple-system,"Segoe UI",Roboto,"Noto Sans JP","Hiragino Sans",sans-serif;color:#0b0b0b;background:#fff;max-width:1000px;margin:2em auto;padding:0 1.5em;line-height:1.7}
h1{font-size:1.6em;border-bottom:2px solid #d9d8d3;padding-bottom:.3em}h2{font-size:1.2em;margin-top:2.2em;color:#2a4a7a;border-left:6px solid #2a78d6;padding-left:.5em}
h3{font-size:1.05em;margin-top:1.4em}p,li{font-size:1em}pre{background:#f4f4f1;border:1px solid #d9d8d3;padding:.7em 1em;overflow-x:auto;font-size:.9em}
code{font-family:ui-monospace,Menlo,Consolas,monospace}table{border-collapse:collapse;margin:.6em 0 1em}th,td{border:1px solid #d9d8d3;padding:.3em .7em;text-align:left;font-size:.95em;vertical-align:top}th{background:#f4f4f1}
.fig{margin:1em auto}.fig svg{width:100%;height:auto;border:1px solid #eee;background:#fff}.cap{font-size:.9em;color:#52514e;margin:-.3em 0 1.4em}
.say{background:#eef5ff;border-left:4px solid #2a78d6;padding:.5em .9em;margin:.6em 0}.say b{color:#2a4a7a}
.note{background:#fff7e6;border-left:4px solid #eda100;padding:.5em .9em;margin:.8em 0}
a{color:#2a4a7a}.meta{color:#52514e;font-size:.85em;margin-top:3em}
.langbar{position:sticky;top:0;background:#fff;padding:.4em 0;border-bottom:1px solid #d9d8d3;z-index:5}
.langbar button{font:inherit;padding:.2em .9em;margin-right:.4em;border:1px solid #2a4a7a;background:#fff;color:#2a4a7a;border-radius:4px;cursor:pointer}
.langbar button.on{background:#2a4a7a;color:#fff}
nav.toc{background:#f4f4f1;border:1px solid #d9d8d3;padding:.6em 1.2em;margin:1em 0}nav.toc ol{margin:.3em 0;padding-left:1.4em}
/* the embedded report (alamode-report output) */
.report{border:2px solid #d9d8d3;padding:0 1.2em 1em;margin-top:1em;background:#fcfcfa;--muted:#59636e;--line:#d0d7de;--head:#f6f8fa;--accent:#0969da;--warn:#9a3412}
.report h1{font-size:1.25em;border:none;margin:.8em 0 .2em}.report h1 small{font-weight:normal;color:var(--muted);font-size:.7em;margin-left:8px}
.report h2{font-size:1.05em;border-left-color:#eda100;color:#0b0b0b;margin-top:1.6em}.report h3{font-size:.95em;color:var(--muted)}
.report p.meta{color:var(--muted);margin:0 0 12px;font-size:.9em}
.report .cards{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}.report .card{flex:1 1 150px;min-width:150px;border:1px solid var(--line);border-radius:8px;padding:8px 12px;background:var(--head)}
.report .card .k{font-size:.8em;color:var(--muted)}.report .card .v{font-size:1.05em;font-weight:600;word-break:break-word}
.report table{font-size:.85em;max-width:100%;display:block;overflow-x:auto}.report th,.report td{white-space:nowrap}.report td.num,.report th.num{text-align:right;font-variant-numeric:tabular-nums}.report td.wrap{white-space:normal;min-width:22em}
.report .warn{color:var(--warn);font-weight:600}.report figure{margin:12px 0 20px}.report figure svg,.report figure img{max-width:100%;height:auto;display:block;background:#fff;border:1px solid #eee;border-radius:6px}
.report figcaption{color:var(--muted);font-size:.85em;margin-top:4px}.report details summary{cursor:pointer;color:var(--accent)}
.report footer{color:var(--muted);font-size:.85em;margin-top:24px;border-top:1px solid var(--line);padding-top:8px}
</style><script>
function setLang(l){document.documentElement.lang=l;
document.querySelectorAll('.lang').forEach(function(e){e.style.display=e.classList.contains(l)?'':'none';});
document.querySelectorAll('.langbar button').forEach(function(b){b.classList.toggle('on',b.dataset.lang===l);});
try{localStorage.setItem('intro_lang',l);}catch(e){}}
document.addEventListener('DOMContentLoaded',function(){var l='ja';try{l=localStorage.getItem('intro_lang')||'ja';}catch(e){}
if(location.hash==='#en')l='en';if(location.hash==='#ja')l='ja';setLang(l);});
</script></head><body>
<div class="langbar"><button data-lang="ja" onclick="setLang('ja')">日本語</button><button data-lang="en" onclick="setLang('en')">English</button></div>
'''

parts = [head]
P = parts.append

P(J('<h1>aiida-alamode 入門 — ALAMODE を AiiDA と LLM から使う</h1>') + E('<h1>Introduction to aiida-alamode — ALAMODE through AiiDA and an LLM</h1>'))

P(J('''<p><b>ALAMODE</b>（alm, anphon）は、変位した超格子の力から調和・非調和の力定数を決め、フォノン分散、状態密度、熱力学量、格子熱伝導率（RTA）、
自己無撞着フォノン（SCPH）、準調和近似（QHA）を計算するプログラムです。<b>aiida-alamode</b> は、その一連の手順を AiiDA のワークフロー基盤の上で動かすためのプラグインです。
変位構造の生成から熱伝導率までの各段を AiiDA のプロセスとして実行し、力は DFT（VASP, Quantum ESPRESSO, OpenMX）、LAMMPS、または機械学習ポテンシャル（ASE 経由、例では MatterSim）から、
LO-TO 分裂に要る Born 有効電荷 Z* と誘電率 ε∞ は手で与えるか機械学習モデル（SevenNet-Polar, Equivar, AnisoNet）から取れます。
計算の投入・監視・結果の取り出し・レポートを CLI と MCP ツールで提供し、Claude のような LLM から日本語で操作できます。</p>
<p>このページは <b>ALAMODE は知っているが AiiDA は知らない人</b>向けです。AiiDA が何を足すのか（来歴）、機械学習モデルで何が省けるのか、レポートと LLM 操作で何が楽になるのかを、実際の計算（MgO, Si）の図で示します。</p>''')
+ E('''<p><b>ALAMODE</b> (alm, anphon) fits harmonic and anharmonic force constants to the forces of displaced supercells and computes phonon dispersions, densities of states,
thermodynamic functions, the lattice thermal conductivity (RTA), self-consistent phonons (SCPH) and the quasi-harmonic approximation (QHA). <b>aiida-alamode</b> is the plugin that runs
this chain on the AiiDA workflow platform. Every step from the displaced structures to the thermal conductivity is an AiiDA process; the forces come from DFT (VASP, Quantum ESPRESSO,
OpenMX), LAMMPS or a machine-learning potential through ASE (the examples use MatterSim), and the Born effective charges Z* and the dielectric constant ε∞ needed for the LO-TO splitting
are either given by hand or predicted by machine-learning models (SevenNet-Polar, Equivar, AnisoNet). Submission, monitoring, results and reports are available as command-line and MCP tools,
so that an LLM such as Claude can operate the plugin in natural language.</p>
<p>This page is for <b>people who know ALAMODE but not AiiDA</b>. It shows what AiiDA adds (provenance), what the machine-learning models save, and what the reports and the LLM
operation make easier, with figures from real calculations (MgO, Si).</p>'''))

P(J('''<nav class="toc"><b>目次</b><ol><li><a href="#arch">構成</a></li><li><a href="#aiida">AiiDA を知らない人へ: ALAMODE の手順はどう写るか</a></li><li><a href="#prov">利点 1: 来歴が残る</a></li>
<li><a href="#bg">背景: 力と Z*・ε∞ の出所</a></li><li><a href="#ml">利点 2: 機械学習モデルで力と Z*・ε∞ を用意する</a></li><li><a href="#report">利点 3: レポートが出せる</a></li>
<li><a href="#llm">利点 4: LLM 経由なので、人にとって操作が楽</a></li><li><a href="#setup">動かすまで</a></li><li><a href="#notes">知っておくこと</a></li><li><a href="#appendix">付録: MgO のレポート</a></li></ol></nav>''')
+ E('''<nav class="toc"><b>Contents</b><ol><li><a href="#arch-en">Architecture</a></li><li><a href="#aiida-en">For AiiDA newcomers: how the ALAMODE steps map</a></li><li><a href="#prov-en">Benefit 1: provenance is kept</a></li>
<li><a href="#bg-en">Background: where the forces, Z* and ε∞ come from</a></li><li><a href="#ml-en">Benefit 2: forces, Z* and ε∞ from machine-learning models</a></li><li><a href="#report-en">Benefit 3: reports</a></li>
<li><a href="#llm-en">Benefit 4: operation through an LLM is easy for people</a></li><li><a href="#setup-en">Getting started</a></li><li><a href="#notes-en">Things to know</a></li><li><a href="#appendix-en">Appendix: the MgO report</a></li></ol></nav>'''))

# ---- architecture
P(J('<h2 id="arch">構成</h2>') + E('<h2 id="arch-en">Architecture</h2>'))
P(J(arch_svg("ja")) + E(arch_svg("en")))
P(J('<p class="cap">図 0: 構成。人は Claude に日本語で指示し、Claude が MCP ツールを呼ぶ。aiida-alamode が AiiDA を通して計算機にジョブを投げ、計算機にインストール済みの alm / anphon と ASE runner（機械学習モデル）が動く。</p>')
+ E('<p class="cap">Fig. 0: architecture. The person instructs Claude; Claude calls MCP tools; aiida-alamode submits jobs through AiiDA to the computer, where the installed alm / anphon and the ASE runner (machine-learning models) run.</p>'))
P(J('''<table><tr><th>層</th><th>役割</th></tr>
<tr><td>Claude + MCP</td><td>「MgO のフォノンを LO-TO 込みで計算して」のような指示を、MCP ツール <code>run_phonons</code> などの呼び出しに変える</td></tr>
<tr><td>aiida-alamode</td><td>ALAMODE の各プログラム（alm suggest / opt / cv, displace, anphon, analyze_phonons）と力・Z*・ε∞ の計算を AiiDA の CalcJob に、BORNINFO の組み立てや力の並列実行を WorkChain にする。driver（<code>example/run_alamode_phonons.py</code> など）が手順を並べ、<code>alamode-report</code> がレポートを書く。MCP <code>alamode-mcp</code></td></tr>
<tr><td>AiiDA</td><td>ジョブの投入（SLURM、ssh）、入出力の保存、<b>来歴（provenance）</b>の記録</td></tr>
<tr><td>ASE runner</td><td><code>alamode-ase-runner</code>：計算機上で ASE の calculator（MatterSim, MACE, CHGNet, SevenNet, ORB …）を動かし、力・緩和・MD・弾性定数を返す。Z* は SevenNet-Polar / Equivar、ε∞ は AnisoNet</td></tr>
<tr><td>ALAMODE</td><td>計算機にインストール済みの alm, displace.py, anphon, analyze_phonons</td></tr></table>''')
+ E('''<table><tr><th>Layer</th><th>Role</th></tr>
<tr><td>Claude + MCP</td><td>turns an instruction such as "compute the phonons of MgO with the LO-TO correction" into calls of MCP tools like <code>run_phonons</code></td></tr>
<tr><td>aiida-alamode</td><td>wraps every ALAMODE program (alm suggest / opt / cv, displace, anphon, analyze_phonons) and the force / Z* / ε∞ calculations as AiiDA CalcJobs, and the BORNINFO assembly and the parallel force runs as WorkChains. Drivers (<code>example/run_alamode_phonons.py</code>, …) arrange the steps and <code>alamode-report</code> writes the report. MCP <code>alamode-mcp</code></td></tr>
<tr><td>AiiDA</td><td>job submission (SLURM, ssh), storage of inputs and outputs, <b>provenance</b></td></tr>
<tr><td>ASE runner</td><td><code>alamode-ase-runner</code>: runs an ASE calculator (MatterSim, MACE, CHGNet, SevenNet, ORB, …) on the computer and returns forces, relaxations, MD and elastic constants. Z* from SevenNet-Polar / Equivar, ε∞ from AnisoNet</td></tr>
<tr><td>ALAMODE</td><td>alm, displace.py, anphon and analyze_phonons installed on the computer</td></tr></table>'''))

# ---- AiiDA for newcomers
P(J('<h2 id="aiida">AiiDA を知らない人へ: ALAMODE の手順はどう写るか</h2>') + E('<h2 id="aiida-en">For AiiDA newcomers: how the ALAMODE steps map</h2>'))
P(J('''<p>AiiDA は計算のワークフロー基盤です。覚えることは 4 つだけです。</p>
<ul><li><b>ノード</b>: 構造、入力ファイル、パラメタ、結果、そして「計算そのもの」もすべてデータベースのノードになり、番号 <b>pk</b> を持つ。</li>
<li><b>リンク</b>: 計算ノードは入力ノードと出力ノードに矢印でつながる。これが<b>来歴（provenance）</b>で、自動で記録される。</li>
<li><b>computer と code</b>: どの計算機のどの実行ファイル（<code>alm@host</code>, <code>anphon@host</code>, <code>ase_runner@host</code>）で走らせるかの登録。SLURM への投入と結果の回収は AiiDA の daemon が行う。</li>
<li><b>verdi</b>: コマンド。<code>verdi process list</code>（走っている計算）、<code>verdi node show &lt;pk&gt;</code>（ノードの中身）、<code>verdi node graph generate &lt;pk&gt;</code>（来歴図）。</li></ul>
<p>ALAMODE のチュートリアルで手で行う手順は、次のように写ります。driver はこれを順に投入し、済んだ段は次回から再利用します（run ディレクトリの <code>.node.json</code> に pk を記録）。</p>
<table><tr><th>ALAMODE の手順（手で行う場合）</th><th>aiida-alamode のプロセス</th></tr>
<tr><td>構造の緩和（DFT）</td><td><code>alamode.relax_ase</code>（体積のみ、または全緩和）</td></tr>
<tr><td><code>alm</code> MODE = suggest</td><td><code>alamode.alm_suggest</code></td></tr>
<tr><td><code>displace.py --VASP … --mag 0.01</code></td><td><code>alamode.displace_pf</code>（有限変位）/ <code>displace_random</code></td></tr>
<tr><td>変位構造ごとに VASP / QE を回す</td><td><code>alamode.forces_ase</code>（MatterSim などで一括）、または自分の DFT / LAMMPS の出力を <code>alamode.extract</code> で DFSET にする</td></tr>
<tr><td><code>alm</code> MODE = optimize（最小二乗、LASSO）</td><td><code>alamode.alm_opt</code> / <code>alm_cv</code></td></tr>
<tr><td>DFPT で Z* と ε∞ を計算して BORNINFO を書く</td><td><code>alamode.borninfo</code>（与えた値、または SevenNet-Polar / Equivar と AnisoNet の予測）</td></tr>
<tr><td><code>anphon</code> phonons / RTA / SCPH / QHA</td><td><code>alamode.anphon</code></td></tr>
<tr><td><code>analyze_phonons.py</code></td><td><code>alamode.analyze_phonons</code></td></tr>
<tr><td>gnuplot / plotband.py</td><td>図の calcfunction（PNG / SVG、summary の Dict 付き）</td></tr></table>''')
+ E('''<p>AiiDA is a workflow platform for calculations. Four things are enough to know.</p>
<ul><li><b>Nodes</b>: structures, input files, parameters, results and the calculations themselves are all nodes in a database, each with a number, the <b>pk</b>.</li>
<li><b>Links</b>: a calculation node is connected by arrows to its input and output nodes. That is the <b>provenance</b>, and it is recorded automatically.</li>
<li><b>Computers and codes</b>: the registration of which executable on which computer runs a step (<code>alm@host</code>, <code>anphon@host</code>, <code>ase_runner@host</code>). The AiiDA daemon submits to SLURM and retrieves the results.</li>
<li><b>verdi</b>: the command line. <code>verdi process list</code> (running calculations), <code>verdi node show &lt;pk&gt;</code> (a node), <code>verdi node graph generate &lt;pk&gt;</code> (the provenance graph).</li></ul>
<p>The steps done by hand in the ALAMODE tutorials map as follows. The drivers submit them in order and reuse the finished steps on the next run (the pks are recorded in <code>.node.json</code> of the run directory).</p>
<table><tr><th>ALAMODE step (by hand)</th><th>aiida-alamode process</th></tr>
<tr><td>relaxation of the structure (DFT)</td><td><code>alamode.relax_ase</code> (volume only, or full)</td></tr>
<tr><td><code>alm</code> MODE = suggest</td><td><code>alamode.alm_suggest</code></td></tr>
<tr><td><code>displace.py --VASP … --mag 0.01</code></td><td><code>alamode.displace_pf</code> (finite displacements) / <code>displace_random</code></td></tr>
<tr><td>VASP / QE for every displaced structure</td><td><code>alamode.forces_ase</code> (all at once with MatterSim etc.), or your own DFT / LAMMPS outputs turned into a DFSET by <code>alamode.extract</code></td></tr>
<tr><td><code>alm</code> MODE = optimize (least squares, LASSO)</td><td><code>alamode.alm_opt</code> / <code>alm_cv</code></td></tr>
<tr><td>DFPT for Z* and ε∞, then write BORNINFO</td><td><code>alamode.borninfo</code> (given values, or predictions of SevenNet-Polar / Equivar and AnisoNet)</td></tr>
<tr><td><code>anphon</code> phonons / RTA / SCPH / QHA</td><td><code>alamode.anphon</code></td></tr>
<tr><td><code>analyze_phonons.py</code></td><td><code>alamode.analyze_phonons</code></td></tr>
<tr><td>gnuplot / plotband.py</td><td>figure calcfunctions (PNG / SVG, with a summary Dict)</td></tr></table>'''))

# ---- benefit 1: provenance
P(J('<h2 id="prov">利点 1: AiiDA が backend なので来歴（provenance）が残る</h2>') + E('<h2 id="prov-en">Benefit 1: AiiDA as the backend keeps the provenance</h2>'))
P(J('''<p>どの構造（CIF）から、どの超格子とカットオフで alm を回し、どの力（どのモデル、何個の変位）から力定数を決め、どの Z*・ε∞ で anphon を回したかが、すべてノードとリンクとして保存されます。
半年後に「この LO の振動数はどの ε∞ で計算した？」と聞かれても、pk ひとつで辿れます。下は MgO の調和力定数の fit（alm opt, pk 14077）の来歴図です
（<code>verdi node graph generate 14077 --ancestor-depth 1 --descendant-depth 1</code>）。上が入力（超格子の構造、DFSET、カットオフ、code）、下が出力（力定数の xml と fcs、結果の Dict）です。</p>''')
+ E('''<p>Which structure (CIF), which supercell and cutoffs ran alm, which forces (which model, how many displacements) gave the force constants, and which Z* and ε∞ went into anphon are all stored as nodes and links.
Asked half a year later "which ε∞ was this LO frequency computed with?", one pk is enough to trace it. Below is the provenance graph of the harmonic fit of MgO (alm opt, pk 14077,
<code>verdi node graph generate 14077 --ancestor-depth 1 --descendant-depth 1</code>): inputs at the top (supercell structure, DFSET, cutoffs, code), outputs at the bottom (force-constant xml and fcs, the results Dict).</p>'''))
P(f'<div class="fig" style="max-width:100%">{graph}</div>')
P(J('<p class="cap">図 1: MgO の alm opt（pk 14077）の来歴。赤が計算、緑がデータ、矢印がリンク。プロセスの種類だけを描いた図（緩和 → alm → displace → 力 → alm → borninfo → anphon → 図）は付録のレポートの末尾にある。</p>')
+ E('<p class="cap">Fig. 1: provenance of the MgO alm opt (pk 14077). Red: calculations, green: data, arrows: links. A graph of the processes only (relax → alm → displace → forces → alm → borninfo → anphon → figures) is at the end of the report in the appendix.</p>'))
P(J('<div class="say"><b>Claude に:</b> 「pk 14182 のフォノン分散はどの力定数と BORNINFO から来たか教えて」 → <code>process_info(pk=14182)</code>（入力リンク: <code>force_constants</code> pk 14077 の xml、<code>borninfo</code> pk 14154）</div>')
+ E('<div class="say"><b>To Claude:</b> "which force constants and BORNINFO did the phonon dispersion of pk 14182 come from?" → <code>process_info(pk=14182)</code> (input links: the xml of <code>force_constants</code> pk 14077, <code>borninfo</code> pk 14154)</div>'))

# ---- background
P(J('<h2 id="bg">背景: 力と Z*・ε∞ の出所</h2>') + E('<h2 id="bg-en">Background: where the forces, Z* and ε∞ come from</h2>'))
P(J('''<p>ALAMODE の入力で時間がかかるのは、変位した超格子の力（調和なら数個、3 次まで含めると数十〜数百個の DFT 計算）と、極性物質で LO-TO 分裂を入れるための Born 有効電荷 Z* と誘電率 ε∞（DFPT）です。
チュートリアルは VASP / QE の出力を前提にしています。aiida-alamode でも自分の DFT / LAMMPS の出力を <code>alamode.extract</code> で読めますが、
例と MCP は既定で機械学習モデルを使います。力は MatterSim（ASE の calculator なら何でも可）、Z* は SevenNet-Polar（Ba, Ca, Hf, Li, O, P, Pb, Sr, Ti, Zr）または Equivar（10 元素）、
ε∞ は AnisoNet（組成と構造から異方性のあるテンソルを予測）です。MgO や NaCl のように Z* の文献値がある物質は <code>--born-charges Mg:1.96 O:-1.96</code> のように手で与え、ε∞ だけ予測することもできます。</p>
<div class="note">機械学習ポテンシャルは DFT ではありません。Si では MatterSim の光学モードは 14.66 THz、チュートリアルの DFT 参照は 15.38 THz、300 K の κ は 150 対 113 W/mK です。
傾向と物理（虚数モードの有無、LO-TO の大きさ、T<sub>c</sub> の目安）を数分で見るための道具で、精密値が要るときは同じワークフローの力の段だけ DFT に置き換えます。</div>''')
+ E('''<p>What takes time in an ALAMODE input are the forces of the displaced supercells (a few DFT runs for the harmonic part, tens to hundreds with the cubic terms) and, for polar materials, the Born
effective charges Z* and the dielectric constant ε∞ for the LO-TO splitting (DFPT). The tutorials assume VASP / QE outputs. aiida-alamode reads your own DFT / LAMMPS outputs too
(<code>alamode.extract</code>), but the examples and the MCP use machine-learning models by default: forces from MatterSim (any ASE calculator works), Z* from SevenNet-Polar (Ba, Ca, Hf, Li, O, P, Pb,
Sr, Ti, Zr) or Equivar (10 elements), ε∞ from AnisoNet (an anisotropic tensor predicted from composition and structure). For materials with literature Z* such as MgO or NaCl, the values are given by hand
(<code>--born-charges Mg:1.96 O:-1.96</code>) and only ε∞ is predicted.</p>
<div class="note">A machine-learning potential is not DFT. For Si the MatterSim optical mode is 14.66 THz against 15.38 THz of the tutorial's DFT reference, and κ at 300 K is 150 versus 113 W/mK.
It is a tool to see the trends and the physics (imaginary modes, size of the LO-TO splitting, a rough T<sub>c</sub>) in minutes; when precise values are needed, only the force step of the same workflow is replaced by DFT.</div>'''))

# ---- benefit 2: ML
P(J('<h2 id="ml">利点 2: 機械学習モデルで力と Z*・ε∞ を用意する</h2>') + E('<h2 id="ml-en">Benefit 2: forces, Z* and ε∞ from machine-learning models</h2>'))
P(J('''<p>MgO（Fm-3m、CIF から）の例です。体積緩和 → 2×2×2 超格子（64 原子）→ alm suggest（2 変位）→ MatterSim の力 → alm opt（fit 誤差 1.1 %）→ Z* は与えた値、ε∞ は AnisoNet（3.21）→ BORNINFO →
anphon を NONANALYTIC = 0 と 3 で実行、の 18 プロセスが CPU で約 1 分で終わります（2026-09-26 17:07:34 → 17:08:32）。図 2 は NA0 と NA3 のバンドと DOS で、Γ 点の最高モードが 11.04 THz（TO）から 20.16 THz（LO）に分裂します。
ε∞ を文献値 3.0 にすると LO は 20.66 THz です。</p>''')
+ E('''<p>An example: MgO (Fm-3m, from a CIF). Volume relaxation → 2×2×2 supercell (64 atoms) → alm suggest (2 displacements) → MatterSim forces → alm opt (fitting error 1.1 %) → given Z*, ε∞ from AnisoNet (3.21) → BORNINFO →
anphon with NONANALYTIC = 0 and 3: 18 processes, about one minute on a CPU (2026-09-26 17:07:34 → 17:08:32). Fig. 2 shows the bands and DOS for NA0 and NA3; the highest Γ mode splits from 11.04 THz (TO) to 20.16 THz (LO).
With the literature ε∞ = 3.0 the LO mode is 20.66 THz.</p>'''))
P(f'<div class="fig" style="max-width:100%">{phband}</div>')
P(J('<p class="cap">図 2: MgO のフォノン分散と DOS（MatterSim の力定数、pk 14077）。左: NONANALYTIC = 0、右: NONANALYTIC = 3（Z* = ±1.96、ε∞ = 3.21 の BORNINFO pk 14154）。<code>--figure-format svg</code> で書いた図の calcfunction の出力そのまま。</p>')
+ E('<p class="cap">Fig. 2: phonon dispersion and DOS of MgO (MatterSim force constants, pk 14077). Left: NONANALYTIC = 0, right: NONANALYTIC = 3 (BORNINFO pk 14154 with Z* = ±1.96, ε∞ = 3.21). The output of the figure calcfunction as written with <code>--figure-format svg</code>.</p>'))
P(J('''<table><tr><th>物質</th><th>Z* の出所</th><th>ε∞ の出所</th><th>Γ 最高モード NA0 → NA3 [THz]</th></tr>
<tr><td>MgO</td><td>文献 1.96</td><td>AnisoNet 3.21</td><td>11.04 → 20.16</td></tr>
<tr><td>NaCl</td><td>文献 1.10</td><td>AnisoNet 2.63</td><td>4.65 → 7.29</td></tr>
<tr><td>γ-Li<sub>3</sub>PO<sub>4</sub>（Pnma, 32 原子）</td><td>SevenNet-Polar（Li 1.04, P 3.04, O −1.54）</td><td>AnisoNet 2.55–2.58</td><td>32.16 → 32.48</td></tr>
<tr><td>PbTe（チュートリアル）</td><td>DFT の BORNINFO</td><td>同</td><td>NA0–3 を DFT 参照と比較</td></tr></table>
<div class="say"><b>Claude に:</b> 「BaHfO3 のフォノンを、Z* と ε∞ も予測して LO-TO 込みで計算して」 → <code>run_phonons(structure="BaHfO3_Pm-3m.cif", supercell=[2,2,2], borninfo_calculator="sevennet-polar", dielectric_model="anisonet", nonanalytic=[0,3], computer="host")</code></div>''')
+ E('''<table><tr><th>Material</th><th>Z* from</th><th>ε∞ from</th><th>highest Γ mode NA0 → NA3 [THz]</th></tr>
<tr><td>MgO</td><td>literature 1.96</td><td>AnisoNet 3.21</td><td>11.04 → 20.16</td></tr>
<tr><td>NaCl</td><td>literature 1.10</td><td>AnisoNet 2.63</td><td>4.65 → 7.29</td></tr>
<tr><td>γ-Li<sub>3</sub>PO<sub>4</sub> (Pnma, 32 atoms)</td><td>SevenNet-Polar (Li 1.04, P 3.04, O −1.54)</td><td>AnisoNet 2.55–2.58</td><td>32.16 → 32.48</td></tr>
<tr><td>PbTe (tutorial)</td><td>DFT BORNINFO</td><td>same</td><td>NA0–3 compared with the DFT reference</td></tr></table>
<div class="say"><b>To Claude:</b> "compute the phonons of BaHfO3 with the LO-TO correction, predicting Z* and ε∞ too" → <code>run_phonons(structure="BaHfO3_Pm-3m.cif", supercell=[2,2,2], borninfo_calculator="sevennet-polar", dielectric_model="anisonet", nonanalytic=[0,3], computer="host")</code></div>'''))

# ---- benefit 3: report
P(J('<h2 id="report">利点 3: レポートが出せる</h2>') + E('<h2 id="report-en">Benefit 3: reports</h2>'))
P(J('''<p>1 つの計算（構造から下流の全プロセス）の要約を、図付きの 1 枚の HTML にできます（<code>alamode-report &lt;run ディレクトリ | 構造の pk&gt;</code>、MCP では <code>run_report</code>、driver は終了時に自動で書きます）。
driver のログではなく AiiDA の来歴から作るので、古い版のプラグインで計算したものや、自分で投入したプロセスにも使えます。内容は、組成式、空間群、Wyckoff 位置、セルの表（入力 → 緩和 → 基本胞 → 超格子）、緩和、力定数の fit、Z* と ε∞、
NONANALYTIC ごとの Γ 点振動数（LO-TO シフト、虚数モードの有無）、熱力学量（零点エネルギー、C<sub>v</sub>, S, F）、RTA の κ(T)、SCPH / QHA の結果、図（SVG は inline）、プロセスの表とグラフ。数値はすべてノードから読み、pk を添えます。
利点 4 の例で作った MgO のレポートを、このページの末尾（付録）に付けてあります。</p>
<div class="say"><b>Claude に:</b> 「MgO の run のレポートを作って」 → <code>run_report(target="…/run_v013/MgO")</code>（HTML のパスと要点、全データの JSON のパスが返る）</div>''')
+ E('''<p>The summary of one calculation (the structure and every process downstream) becomes one HTML file with figures (<code>alamode-report &lt;run directory | structure pk&gt;</code>, <code>run_report</code> in the MCP; the drivers write it at the end).
It is built from the AiiDA provenance, not from the driver's log, so it also works for runs of older plugin versions and for processes you submitted yourself. Contents: formula, space group, Wyckoff positions, the cells (input → relaxed → primitive → supercell),
the relaxation, the force-constant fits, Z* and ε∞, the Γ frequencies per NONANALYTIC value (LO-TO shift, imaginary modes), thermodynamics (zero-point energy, C<sub>v</sub>, S, F), the RTA κ(T), SCPH / QHA results, the figures (SVG inline),
the table and graph of the processes. Every number is read from a node and carries its pk. The MgO report made in the example of benefit 4 is attached at the end of this page (appendix).</p>
<div class="say"><b>To Claude:</b> "make the report of the MgO run" → <code>run_report(target="…/run_v013/MgO")</code> (returns the HTML path, the key numbers and the path of the JSON with all collected data)</div>'''))

# ---- benefit 4: LLM
P(J('<h2 id="llm">利点 4: LLM 経由なので、人にとって操作が楽</h2>') + E('<h2 id="llm-en">Benefit 4: operation through an LLM is easy for people</h2>'))
P(J('''<p>alm / anphon の入力の書式、AiiDA の verdi コマンド、pk の追い方を覚えなくても、やりたいことを日本語で言えば Claude が対応する MCP ツールを選び、順に呼び、結果を読んで説明します。
フォノン計算は 10 段以上のプロセスの連鎖で、途中に小さな判断（超格子の大きさ、カットオフ、Z*・ε∞ の出所、alm が「自由な力定数 0 個」と言ったとき、anphon に虚数モードが出たとき）があります。
プラグインの skill（<code>.claude/skills/aiida-alamode/SKILL.md</code>）にその判断と落とし穴が書いてあるので、エージェントが自分で対処します。
ツールは読み取り（<code>run_status</code>, <code>run_results</code>, <code>list_runs</code>, <code>process_info</code>, <code>run_report</code>）と投入（<code>run_phonons</code>, <code>run_driver</code>, <code>kill_run</code>）に分かれています。</p>
<h3>実際の指示例</h3>
<p><b>人:</b> 「<code>MgO_Fm-3m.cif</code> を読んで、Z* は Mg 1.96 / O −1.96、ε∞ は AnisoNet で、LO-TO 込みのフォノンを計算してレポートを出せ」</p>
<p>Claude はこれを次の段に分けて実行します（2026-09-26 に行った例。pk はそのときの値）。</p>
<table><tr><th>段</th><th>Claude が呼ぶツール</th><th>返るもの</th></tr>
<tr><td>1. 計算機のパッケージを確認</td><td><code>check_packages(computer="host")</code></td><td>mattersim, anisonet: ok（sevennet-polar は不要）</td></tr>
<tr><td>2. 投入</td><td><code>run_phonons(structure="MgO_Fm-3m.cif", supercell=[2,2,2], name="MgO", born_charges={"Mg":[1.96],"O":[-1.96]}, dielectric_model="anisonet", nonanalytic=[0,3], computer="host")</code></td><td>run_id、ログのパス</td></tr>
<tr><td>3. 待つ</td><td><code>run_status(run_id)</code> を繰り返す</td><td>段ごとの状態（relax 13917 → alm 13960 → displace 13998 → forces 14025 → alm opt 14077 → borninfo 14104 → anphon 14165 / 14182 → 図）</td></tr>
<tr><td>4. 結果を読む</td><td><code>run_results(run_dir)</code></td><td>緩和後 a = 4.254 Å、fit 誤差 1.12 %、Z* = ±1.96（与えた値）、ε∞ = 3.21（AnisoNet）、Γ 最高 11.04 → 20.16 THz、虚数モード無し、ZPE 140 meV、すべて pk 付き</td></tr>
<tr><td>5. レポート</td><td><code>run_report(target=run_dir)</code></td><td><code>MgO_report.html</code>（付録）と要点、<code>MgO_report.json</code></td></tr></table>
<h3>他の言い方</h3>
<table><tr><th>人の言葉</th><th>Claude が呼ぶもの</th></tr>
<tr><td>「Si のフォノンと熱伝導率をチュートリアルと比べて」</td><td><code>run_driver(driver="phonons", args=["--preset","Si"])</code></td></tr>
<tr><td>「BaTiO3 の SCPH で T<sub>c</sub> を見て」</td><td><code>run_driver(driver="scph", args=[…])</code></td></tr>
<tr><td>「今走っている計算は？」</td><td><code>list_runs()</code>, <code>run_status(run_id)</code></td></tr>
<tr><td>「pk 14077 の fit の中身を見せて」</td><td><code>process_info(pk=14077)</code></td></tr>
<tr><td>「BaZrO3 の R 点は不安定？」</td><td><code>run_phonons(…)</code> → <code>run_results</code> の <code>imaginary_modes</code></td></tr></table>
<p>全部の操作は CLI（<code>example/run_alamode_phonons.py</code>, <code>run_alamode_scph.py</code>, <code>run_alamode_qha.py</code>, <code>alamode-report</code>）でも同じにできます。MCP は driver をサブプロセスで起動する薄い層で、サーバを止めても AiiDA の計算は続きます。</p>''')
+ E('''<p>Without learning the alm / anphon input format, the verdi commands of AiiDA or how to follow pks, you say what you want and Claude picks the MCP tools, calls them in order, reads the results and explains them.
A phonon calculation is a chain of ten or more processes with small decisions on the way (supercell size, cutoffs, the source of Z* and ε∞, what to do when alm reports "0 free force constants" or anphon shows an imaginary mode).
The plugin's skill (<code>.claude/skills/aiida-alamode/SKILL.md</code>) holds those decisions and pitfalls, so the agent handles them itself.
The tools are split into reading (<code>run_status</code>, <code>run_results</code>, <code>list_runs</code>, <code>process_info</code>, <code>run_report</code>) and submitting (<code>run_phonons</code>, <code>run_driver</code>, <code>kill_run</code>).</p>
<h3>A real instruction</h3>
<p><b>Person:</b> "read <code>MgO_Fm-3m.cif</code>, take Z* = Mg 1.96 / O −1.96 and ε∞ from AnisoNet, compute the phonons with the LO-TO correction and write the report"</p>
<p>Claude splits this into the following steps (done on 2026-09-26; the pks are those of that run).</p>
<table><tr><th>Step</th><th>Tool Claude calls</th><th>What comes back</th></tr>
<tr><td>1. check the computer's packages</td><td><code>check_packages(computer="host")</code></td><td>mattersim, anisonet: ok (sevennet-polar not needed)</td></tr>
<tr><td>2. submit</td><td><code>run_phonons(structure="MgO_Fm-3m.cif", supercell=[2,2,2], name="MgO", born_charges={"Mg":[1.96],"O":[-1.96]}, dielectric_model="anisonet", nonanalytic=[0,3], computer="host")</code></td><td>run_id, log path</td></tr>
<tr><td>3. wait</td><td><code>run_status(run_id)</code> repeatedly</td><td>the state of each step (relax 13917 → alm 13960 → displace 13998 → forces 14025 → alm opt 14077 → borninfo 14104 → anphon 14165 / 14182 → figures)</td></tr>
<tr><td>4. read the results</td><td><code>run_results(run_dir)</code></td><td>relaxed a = 4.254 Å, fitting error 1.12 %, Z* = ±1.96 (given), ε∞ = 3.21 (AnisoNet), highest Γ mode 11.04 → 20.16 THz, no imaginary modes, ZPE 140 meV, every value with its pk</td></tr>
<tr><td>5. report</td><td><code>run_report(target=run_dir)</code></td><td><code>MgO_report.html</code> (appendix), the key numbers, <code>MgO_report.json</code></td></tr></table>
<h3>Other phrasings</h3>
<table><tr><th>What the person says</th><th>What Claude calls</th></tr>
<tr><td>"compare the phonons and thermal conductivity of Si with the tutorial"</td><td><code>run_driver(driver="phonons", args=["--preset","Si"])</code></td></tr>
<tr><td>"look at T<sub>c</sub> of BaTiO3 with SCPH"</td><td><code>run_driver(driver="scph", args=[…])</code></td></tr>
<tr><td>"what is running now?"</td><td><code>list_runs()</code>, <code>run_status(run_id)</code></td></tr>
<tr><td>"show me the fit of pk 14077"</td><td><code>process_info(pk=14077)</code></td></tr>
<tr><td>"is the R point of BaZrO3 unstable?"</td><td><code>run_phonons(…)</code> → <code>imaginary_modes</code> of <code>run_results</code></td></tr></table>
<p>Everything can be done the same way from the command line (<code>example/run_alamode_phonons.py</code>, <code>run_alamode_scph.py</code>, <code>run_alamode_qha.py</code>, <code>alamode-report</code>). The MCP is a thin layer that launches the drivers as subprocesses; the AiiDA calculations keep running when the server stops.</p>'''))

# ---- setup
P(J('<h2 id="setup">動かすまで</h2>') + E('<h2 id="setup-en">Getting started</h2>'))
setup_code = '''# 1. ALAMODE と AiiDA（必須）。alm, anphon, displace.py, analyze_phonons をビルドし、AiiDA のプロファイルを作る
pip install -e .                     # aiida-core, ase, spglib, numpy, pandas, matplotlib
verdi presto                         # プロファイル（PostgreSQL + RabbitMQ、または verdi presto の sqlite）
verdi computer setup … ; verdi code create core.code.installed --label alm --computer host …   # anphon, displace, analyze_phonons, ase_runner も
verdi daemon start

# 2. 機械学習モデル（任意。code ase_runner が走る計算機に入れる）
pip install -e .[mattersim]          # 力
pip install -e .[sevennet-polar]     # Z*（または .[equivar]）
pip install -e <AnisoNet の clone>    # ε∞
python example/check_packages.py --computer host     # 何が入っているか

# 3. 試す
cd example
python run_alamode_phonons.py --preset Si --computer host           # チュートリアルの Si（DFT 参照との比較つき）
python run_alamode_phonons.py --structure MgO_Fm-3m.cif --supercell 2 2 2 --name MgO \\
    --born-charges Mg:1.96 O:-1.96 --dielectric-model anisonet --nonanalytic 0 3 --computer host

# 4. LLM から使う
pip install -e .[mcp]
claude mcp add --scope user --transport stdio aiida-alamode -- alamode-mcp     # または repo 内で claude を起動（.mcp.json）'''
setup_code_en = setup_code.replace("# 1. ALAMODE と AiiDA（必須）。alm, anphon, displace.py, analyze_phonons をビルドし、AiiDA のプロファイルを作る", "# 1. ALAMODE and AiiDA (required): build alm, anphon, displace.py, analyze_phonons and make an AiiDA profile") \
    .replace("# プロファイル（PostgreSQL + RabbitMQ、または verdi presto の sqlite）", "# a profile (PostgreSQL + RabbitMQ, or the sqlite one of verdi presto)") \
    .replace("# anphon, displace, analyze_phonons, ase_runner も", "# the same for anphon, displace, analyze_phonons, ase_runner") \
    .replace("# 2. 機械学習モデル（任意。code ase_runner が走る計算機に入れる）", "# 2. machine-learning models (optional; on the computer where the code ase_runner runs)") \
    .replace("# 力", "# forces").replace("# Z*（または .[equivar]）", "# Z* (or .[equivar])").replace("<AnisoNet の clone>    # ε∞", "<clone of AnisoNet>   # ε∞") \
    .replace("# 何が入っているか", "# what is installed").replace("# 3. 試す", "# 3. try it").replace("# チュートリアルの Si（DFT 参照との比較つき）", "# the tutorial's Si (compared with the DFT reference)") \
    .replace("# 4. LLM から使う", "# 4. from an LLM").replace("# または repo 内で claude を起動（.mcp.json）", "# or start claude inside the repo (.mcp.json)")
P(J(f'<pre>{html.escape(setup_code)}</pre><p>詳しくは README（install, quick start）、<code>docs/remote_gpu_computer.md</code>（ssh + SLURM の計算機）、<code>docs/mcp_server.md</code>。チュートリアルの preset（Si, PbTe, BaTiO3, ZnO）は ALAMODE 1.5.0 の <code>example/</code> を repo の隣に <code>alamode_test/</code> として置くと DFT 参照と比較します。</p>')
+ E(f'<pre>{html.escape(setup_code_en)}</pre><p>Details: README (install, quick start), <code>docs/remote_gpu_computer.md</code> (a computer over ssh + SLURM), <code>docs/mcp_server.md</code>. The tutorial presets (Si, PbTe, BaTiO3, ZnO) compare with the DFT reference when the <code>example/</code> of ALAMODE 1.5.0 is placed next to the repository as <code>alamode_test/</code>.</p>'))

# ---- notes
P(J('<h2 id="notes">知っておくこと</h2>') + E('<h2 id="notes-en">Things to know</h2>'))
P(J('''<ul>
<li><b>超格子の像</b>: <code>ase.build.make_supercell</code> は原子を別の周期像に置くことがある（PbTe 4×4×4 の Te）。調和フォノンは合うが anphon の NONANALYTIC = 3（Ewald）が大きく狂う。プラグインは <code>make_diagonal_supercell</code>（x<sub>prim</sub> + T）で作る。</li>
<li><b>BORNINFO の原子順</b>: anphon の KD は元素番号順（例 Te Pb）、BORNINFO は &amp;position の順。プラグインが揃えるが、自分で書いた BORNINFO を渡すときは注意。</li>
<li><b>「自由な力定数 0 個」</b>: 全緩和したセルの 1e-6 程度のずれで alm の対称性判定が落ちる。<code>--idealize</code>（spglib で対称化）で直る。</li>
<li><b>虚数モード</b>: BaZrO3 の R 点回転のように物理的なものもある（SCPH で温度依存を見る）。<code>run_results</code> の <code>imaginary_modes</code> で分かる。</li>
<li><b>基本胞の向き</b>: spglib の find_primitive は格子を回す。anphon の基本胞は超格子と同じ直交座標系でなければならず、ずれるとバンドが周期的でなくなる（X 点で跳ぶ）。</li>
<li><b>長い WorkChain</b>: aiida-core 2.9 の RabbitMQ の consumer timeout（<code>docs/rabbitmq.md</code>）。</li>
<li><b>GPU</b>: 無くても動く（MatterSim の力は CPU で数秒〜数十秒）。5000 ステップの MD などは GPU が数倍速い（<code>--gpu</code>）。</li></ul>''')
+ E('''<ul>
<li><b>Supercell images</b>: <code>ase.build.make_supercell</code> may put atoms at another periodic image (Te in the 4×4×4 PbTe). Harmonic phonons are fine, but anphon's NONANALYTIC = 3 (Ewald) goes badly wrong. The plugin builds supercells as x<sub>prim</sub> + T (<code>make_diagonal_supercell</code>).</li>
<li><b>Atom order of BORNINFO</b>: anphon's KD is in the order of the atomic number (e.g. Te Pb), BORNINFO follows &amp;position. The plugin lines them up; take care when passing your own BORNINFO.</li>
<li><b>"0 free force constants"</b>: a fully relaxed cell with 1e-6 noise fails alm's symmetry test. <code>--idealize</code> (spglib symmetrisation) fixes it.</li>
<li><b>Imaginary modes</b>: some are physical, like the R-point rotation of BaZrO3 (SCPH shows the temperature dependence). <code>imaginary_modes</code> of <code>run_results</code> tells.</li>
<li><b>Orientation of the primitive cell</b>: spglib's find_primitive rotates the lattice. anphon's primitive cell must be in the same Cartesian frame as the supercell, otherwise the bands are not periodic (a jump at X).</li>
<li><b>Long WorkChains</b>: the RabbitMQ consumer timeout of aiida-core 2.9 (<code>docs/rabbitmq.md</code>).</li>
<li><b>GPU</b>: not needed (MatterSim forces take seconds to tens of seconds on a CPU). Long MD runs (5000 steps) are several times faster on a GPU (<code>--gpu</code>).</li></ul>'''))

# ---- appendix
P(J('<h2 id="appendix">付録: MgO のレポート（利点 4 の例の出力）</h2><p>「MgO の run のレポートを作って」（<code>run_report</code>、= <code>alamode-report example/run_v013/MgO</code>）で得られたファイルをそのまま載せています（計算機名は host に置き換え）。</p>')
+ E('<h2 id="appendix-en">Appendix: the MgO report (output of the example of benefit 4)</h2><p>The file obtained by "make the report of the MgO run" (<code>run_report</code>, = <code>alamode-report example/run_v013/MgO</code>), as it is (the computer name replaced by host).</p>'))
P(f'<div class="report">{report_main}</div>')

P(J('<p class="meta">aiida-alamode 1.0.0（2026-09-26）。図は inline SVG（来歴図は AiiDA の graphviz 出力、その他は matplotlib の calcfunction）。</p>')
+ E('<p class="meta">aiida-alamode 1.0.0 (2026-09-26). Figures are inline SVG (the provenance graph from the graphviz output of AiiDA, the others from the matplotlib calcfunctions).</p>'))
P('</body></html>\n')

html_out = "\n".join(parts)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html_out)
print(OUT, len(html_out), "chars")
