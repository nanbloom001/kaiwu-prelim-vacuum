# Monitor Config Note

## Do Not Change `server_req_base_url`

In `train/.docker-compose.yaml`, keep this exact setting:

```yaml
server_req_base_url: http://127.0.0.1:${MONITOR_TRPC_PORT}
```

This is the correct local monitor configuration.

Reason:

- The monitor page is served by `fe-monitor-service`, but the page runs API requests from the user's browser on the host machine.
- `127.0.0.1:${MONITOR_TRPC_PORT}` resolves from the host browser to Docker's published port, for example `127.0.0.1:11001 -> monitor-service:8040`.
- `http://monitor-service:8040` is only resolvable inside the Docker network. The host browser cannot resolve the Docker service name `monitor-service`, so the web panel reports `Fail to fetch`.

Operational rule:

- Do not replace `server_req_base_url` with `http://monitor-service:8040`.
- If the monitor panel fails, first recreate `fe-monitor-service` with the current compose file:

```bash
cd train
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed up -d --force-recreate fe-monitor-service
```

Related historical note:

- `train/context/LOG_20260413_resume_fix_and_local_training.md`, section `server_req_base_url 注意事项`.
