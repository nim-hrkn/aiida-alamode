# RabbitMQ（AiiDA のブローカ）：版の確認と consumer timeout

AiiDA の daemon は RabbitMQ 経由でプロセスを受け取る。RabbitMQ 3.8.15 以降には **consumer timeout**
（消費者が ack を返さないまま経過できる時間、既定 30 分）があり、これより長く走る AiiDA のプロセス
（数時間の MD、大きな胞の anphon、GPU ノードの待ち行列）は途中で接続を切られて
"Consumer timeout" / "Process ... was terminated" になる。AiiDA の推奨どおり値を伸ばしておく。

## 版の確認

RabbitMQ は conda の `alamode` 環境（conda-forge の `rabbitmq-server`）にユーザ権限で入っている。

```
verdi status                                   # broker: RabbitMQ v4.3.4 @ amqp://guest:***@127.0.0.1:5672?heartbeat=600
~/miniforge3/envs/alamode/bin/rabbitmqctl version          # 4.3.4
conda list -n alamode | grep -E 'rabbitmq|erlang'          # rabbitmq-server 4.3.4, erlang 27.3.4
```

`verdi status` が broker の版を出すので、普段はこれで足りる。

**対応する版**：AiiDA の公式の動作確認は RabbitMQ 3 系（3.8〜3.13）だが、ここでは **4.3.4 でも動作する**ことを確認している
（AiiDA 2.9.2、2026-09-22 以降。MatterSim の CalcJob、WorkChain、`core.ssh_async` の GPU ノード、数百件のプロセスを
問題なく処理）。4 系でも 3.8.15 以降と同じく consumer timeout の設定が要る点は変わらない。
`verdi status` が "incompatible RabbitMQ version" の警告を出す場合は、
`verdi config set warnings.rabbitmq_version False` で警告だけ止める（動作は変わらない）。

## consumer timeout を 6 日にする

設定ファイルは環境の `etc/rabbitmq/rabbitmq.conf`（ミリ秒）。6 日 = 6 × 24 × 3600 × 1000 = 518 400 000 ms。

```
# ~/miniforge3/envs/alamode/etc/rabbitmq/rabbitmq.conf
listeners.tcp.default = 127.0.0.1:5672
management.tcp.ip = 127.0.0.1
# AiiDA: consumer timeout 6 days (ms); longer than any CalcJob / WorkChain here
consumer_timeout = 518400000
```

6 日にする理由：slurm の `max_wallclock_seconds`（既定 1 時間、長いものでも 1 日）と、GPU ノードの待ち行列・
ssh の再試行を足しても 1 週間を超えないため。`undefined`（無効化）は RabbitMQ 4 系では設定ファイルからは
書けないので、有限の大きな値にする。

この値は 2026-09-26 に反映済み（それ以前は 36 000 000 000 ms ≈ 417 日だった）。

反映の順番（daemon を先に止める。逆だと daemon が接続を失って processes が `Waiting` のまま残る）：

```
verdi daemon stop
~/miniforge3/envs/alamode/bin/rabbitmqctl stop
~/miniforge3/envs/alamode/bin/rabbitmq-server -detached
sleep 5
~/miniforge3/envs/alamode/bin/rabbitmqctl environment | grep consumer_timeout   # {consumer_timeout,518400000}
verdi daemon start
```

`rabbitmqctl environment` の値が設定ファイルと一致していれば反映されている。RabbitMQ は起動時に自動では立ち上がらないので、
再起動後は `rabbitmq-server -detached` → `verdi daemon start` の順に手で上げる。ログは環境の
`var/log/rabbitmq/rabbit@<host>.log`。

## 関連

- [remote_gpu_computer.md](remote_gpu_computer.md)：GPU ノードのジョブは 30〜60 秒の ssh / slurm オーバヘッドがあり、
  待ち行列によっては数時間 `Waiting` になる。consumer timeout はその間も切れない長さが要る。
- `verdi process list -a -p 1` で `Waiting` のまま古いものが残っていれば、timeout で切られた可能性がある。`verdi process play <pk>` で再開する。
