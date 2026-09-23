# slurm のジョブが Reason=InvalidAccount で待たされる（slurm-wlm 23.11.4、accounting_storage/none）

## 症状

- 会計を使わない設定（`AccountingStorageType=accounting_storage/none`、`AccountingStorageEnforce` 未設定、slurmdbd なし）
  なのに、投入したジョブが `squeue` で `(InvalidAccount)` のまま待つ。
- slurmctld のログに、スケジューラの周期ごとに次の 2 行が対で出る。
  ```
  error: _refresh_assoc_mgr_qos_list: no new list given back keeping cached one.
  sched: JobId=NNN has invalid account
  ```
- ジョブは結局は動く。ただし起動するのはバックフィルだけで、`bf_interval`（既定 30 秒）に 1 本ずつ。
  行列が長いと 1 本あたり数分待つ。ノードが idle でも同じ。
- Ubuntu 24.04 の slurm-wlm 23.11.4-1.2ubuntu5 で、2 台の別の計算機で同じ症状を確認した（2026-09-23）。

## 原因（slurm のソースで確認）

1. slurm 23.11 では `accounting_storage/none` は実プラグインではなく NOOP（`src/interfaces/accounting_storage.c` で
   `plugin_inited = PLUGIN_NOOP`）。`acct_storage_g_get_connection()` は NULL を返し、errno には触らない。
2. slurmctld は起動時に `assoc_mgr_init(acct_db_conn, &assoc_init_arg, errno)` と**その時点の errno をそのまま渡す**
   （`src/slurmctld/controller.c` の `ctld_assoc_mgr_init()`、23.11 でも master でも同じ）。errno は直前までに残った値で、
   無くてもよい設定ファイルの探索（/etc/slurm/cgroup.conf、plugstack.conf が無いと ENOENT）や、MPI プラグインの
   読み込み失敗（mpi/pmix で libpmix が無い）などで非ゼロになっている。
3. `assoc_mgr_init()` は `db_conn_errno != SLURM_SUCCESS` なら association のリストを作らずに SLURM_ERROR を返す
   （`src/common/assoc_mgr.c`）。slurmctld は「DB が落ちているので状態ファイルから読む」経路に入り、リストは NULL のまま。
4. メインスケジューラ `_schedule()` は各ジョブを `assoc_mgr_validate_assoc_id()` で検査する（`src/slurmctld/job_scheduler.c`）。
   リストが NULL なので `assoc_mgr_refresh_lists()` を呼び、NOOP から QOS リストが返らず失敗 → "has invalid account" →
   `state_reason = FAIL_ACCOUNT`（= InvalidAccount）。この検査はバックフィルには無いので、バックフィルだけが起動する。

つまり設定の残骸ではなく、「NOOP の会計プラグイン + 未初期化の errno」という slurmctld 側の不具合。
状態ファイル（StateSaveLocation の assoc_mgr_state）を消しても直らない。

## 直し方（いずれも root）

### A. 実際の会計ストアを使う（推奨）

slurmdbd + MariaDB を立てて `accounting_storage/slurmdbd` にする。接続が本物になり association のリストが作られて
InvalidAccount は消える。副産物として `sacct` が使えるようになり、AiiDA の `detailed_job_info` にジョブの経過時間と
CPU 時間が入る。

```
apt install slurmdbd mariadb-server
mysql -e "CREATE DATABASE slurm_acct_db; CREATE USER 'slurm'@'localhost' IDENTIFIED BY '<pass>'; GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';"
# /etc/slurm/slurmdbd.conf (owner slurm, mode 600)
#   DbdHost=localhost  StorageType=accounting_storage/mysql  StorageUser=slurm  StoragePass=<pass>  StorageLoc=slurm_acct_db
#   AuthType=auth/munge  SlurmUser=slurm  LogFile=/var/log/slurm/slurmdbd.log  PidFile=/run/slurmdbd.pid
# /etc/slurm/slurm.conf
#   AccountingStorageType=accounting_storage/slurmdbd
#   AccountingStorageHost=localhost
systemctl enable --now slurmdbd
sacctmgr -i add cluster <ClusterName>          # slurm.conf の ClusterName と同じ
sacctmgr -i add account default cluster=<ClusterName>
sacctmgr -i add user <user> account=default
systemctl restart slurmctld
```

### B. slurmctld に 1 行のパッチを当てる

`src/slurmctld/controller.c` の `ctld_assoc_mgr_init()` で、`assoc_mgr_init(acct_db_conn, &assoc_init_arg, errno)` の
直前に `errno = 0;` を入れる（NOOP のとき errno に意味はない）。Ubuntu なら `apt source slurm-wlm` → パッチ →
`dpkg-buildpackage` → slurmctld のパッケージだけ入れ替え。手間は大きい。

### C. 緩和（設定だけ、InvalidAccount は残る）

`/etc/slurm/slurm.conf` に
```
SchedulerParameters=bf_interval=5,bf_continue,bf_max_job_test=200
```
を足して `scontrol reconfigure`。バックフィルの周期が 30 秒から 5 秒になり、待ちが 1/6 になる。

### 効かないこと

- `AccountingStorageEnforce` の変更（検査はその前で失敗している）。
- assoc_mgr_state の削除。
- mpi/pmix の警告だけを消す（`apt install libpmix2t64`）。errno の発生源はほかにもある。

## 直ったかの確認

- `journalctl -u slurmctld -f` に "has invalid account" が出なくなる。
- `squeue` の Reason が InvalidAccount でなく Resources / Priority / None になる。
- A の場合は `sacct` にジョブが記録される。

## AiiDA 側の留意点

InvalidAccount のままでもジョブは進むので、AiiDA の core.slurm プラグインは正常に扱える（状態 PENDING → RUNNING → 終了）。
遅いのは slurm 側の起動間隔であり、AiiDA のポーリング間隔（`minimum_scheduler_poll_interval`）を短くしても効果は小さい。
