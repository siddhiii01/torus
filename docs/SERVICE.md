# Service (P11.1, P11.2)

Install: `pip install -e ".[service]"`. Set `TORUS_DATA_DIR` and `TORUS_API_TOKEN`
(at least 16 characters; the API refuses to start without it). Settings live in
`config/service.yaml`.

```bash
python -m torus.service serve --port 8080          # API + inbox folder watcher
python -m torus.service worker \
  --runner torus.service.pipeline_runner:build_runner [--with-model-servers]
```

Run as many `worker` processes as needed; they share the SQLite queue in
`TORUS_DATA_DIR/service/`. A worker that dies is recovered when its lease expires.

| Endpoint | Purpose |
|---|---|
| `POST /v1/jobs` `{"path": "<relative to TORUS_DATA_DIR>", "template_id"?}` | queue a file or folder (one job per file, one batch id) |
| `PUT /v1/jobs/upload?suffix=pdf` (raw body) | upload one file |
| `GET /v1/jobs/{id}`, `GET /v1/batches/{id}` | status (metadata only) |
| `GET /v1/jobs/{id}/result` | download the Excel |
| `POST /v1/jobs/{id}/cancel` | cancel a queued job |
| `GET /healthz` | liveness and queue length (no auth) |

All `/v1` routes need `Authorization: Bearer <token>`.

Stages (`extract`, `write`) each have a timeout and retry policy in
`config/service.yaml` (`retry`, `stages.<name>`). A stage that runs out of
attempts fails the job with `error_code = "<stage>:<ExceptionType>"`; messages
are never stored. Model servers are separate processes listed under
`model_servers` (health URLs must be loopback/internal).

## Audit, monitoring, security (P11.3 to P11.5)

* **Audit log** (`audit_events` table, tamper-evident hash chain): per job it
  records submission, start, each stage with timing and attempt, success (config
  hash, model versions, timings, status counts) or failure (error code), and
  spot checks. Only allow-listed metadata keys are accepted. `GET
  /v1/jobs/{id}/audit` adds reviewer actions (who, which column, what action; no
  values). `python -m torus.service verify-audit` checks the chain.
* **Monitoring**: `GET /v1/metrics` returns queue length, throughput, review
  rate, median job time, model health and the silent-error rate from spot checks.
  `GET /v1/spot-checks/pending` lists sampled jobs (rate in `config/service.yaml`)
  whose confirmed values a person should re-check; answer with `POST
  /v1/jobs/{id}/spot-check`.
* **Retention**: `python -m torus.service purge` (run it from a scheduler)
  deletes job work files and uploaded copies older than `retention.job_days`;
  `DELETE /v1/jobs/{id}` does it now. Inbox files are never deleted.
* **No outbound network from workers**: workers install an in-process guard that
  refuses non-loopback, non-private connections. This is defence in depth: also
  block egress at the container/firewall (P11.6).

Not done: encryption at rest for `TORUS_DATA_DIR` (needs a decision, see below),
TLS (terminate it at a reverse proxy), Docker packaging (P11.6).
