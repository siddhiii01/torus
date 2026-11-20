# Operations: deploy, run, review, upgrade, roll back

Companion to `docs/SERVICE.md`. This is the procedure an operator follows.

## 1. Images

```bash
docker build -f deploy/Dockerfile.cpu -t torus:cpu .
docker build -f deploy/Dockerfile.gpu -t torus:gpu .      # NVIDIA hosts
```

Optional model stacks: `--build-arg EXTRAS=service,paddle,doctr` (CPU) or
`EXTRAS=service,vlm` and `--build-arg TORCH_INDEX_URL=<wheel index>` (GPU).
Match the torch/paddle build to the host driver (see `docs/SETUP.md`). Model
weights are never in the image: mount them at `/models` (`TORUS_MODEL_DIR`).

The Dockerfiles have **not** been built in CI yet; do a trial build on the
target host before the first deployment.

## 2. First deploy

1. `cp deploy/.env.example deploy/.env` and set `TORUS_API_TOKEN` (16+ chars) and
   `TORUS_IMAGE`.
2. Enable disk encryption for the volume that holds `/data` (BitLocker, LUKS or
   the cloud provider's volume encryption). The app does not encrypt files itself.
3. Install the config bundle (section 3), then:
   `docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d`.
4. Check `curl http://127.0.0.1:8080/healthz` and `GET /v1/metrics`
   (with the bearer token).

The worker runs on an `internal` network with no route out, and as a non-root
user with a read-only filesystem. Terminate TLS at a reverse proxy in front of
port 8080.

## 3. Config bundles

One bundle = one versioned, hash-verified copy of `config/` (templates, page
types, readers, pipeline, service). Same content always gives the same id.

```bash
python -m torus.service bundle create --config-dir config --out torus-cfg.tgz   # validates first
python -m torus.service bundle install --root /var/lib/torus/bundles torus-cfg.tgz
python -m torus.service bundle activate --root /var/lib/torus/bundles <bundle_id>
python -m torus.service bundle status   --root /var/lib/torus/bundles
```

In the containers use `docker compose exec api python -m torus.service bundle
... --root /var/lib/torus/bundles`. The entrypoint reads the active bundle on
start, so activating takes effect after `docker compose restart api worker`.
Every job's audit record carries the `config_hash` that ran.

## 4. Upgrade

1. Take the queue to a stable point: stop submissions (or accept that running
   jobs are re-queued when their lease expires).
2. Back up the `torus-data` volume (`service/jobs.sqlite3` holds the queue and
   audit log).
3. Install the new bundle; **do not activate yet**. Run
   `python -m torus.service verify-audit` to confirm the audit chain is intact.
4. `docker pull`/build the new image, set `TORUS_IMAGE`, activate the new
   bundle, then `docker compose up -d`.
5. Smoke test: submit a known synthetic pack, confirm `succeeded`, check
   `/v1/metrics` and `model_health`.

## 5. Roll back

```bash
python -m torus.service bundle rollback --root /var/lib/torus/bundles   # previous config
# set TORUS_IMAGE back to the previous tag
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d
```

Rollback only moves a pointer, so it is instant and old bundles remain
installed. Jobs that finished under the newer version keep their results and
audit records. The job database schema has not changed since the first release;
if a future release changes it, that release's notes must say how to restore the
backup from step 2.

## 6. Daily operation

- **Run**: drop files in the inbox (watcher) or use the API; results at
  `GET /v1/jobs/{id}/result`.
- **Review**: values marked `needs_review` go through `torus review`; corrections
  are stored and appear (without values) in the job's audit report.
- **Spot checks**: work through `GET /v1/spot-checks/pending` and record each
  outcome; watch `silent_error_rate` in `/v1/metrics` (target 0).
- **Retention**: schedule `python -m torus.service purge` (daily).
