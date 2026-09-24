# ModelXray container deployment

The production-oriented compose file is intentionally conservative:
- API runs as a non-root user.
- API filesystem is read-only except `/app/data` and `/tmp`.
- Linux container drops all capabilities and enables `no-new-privileges`.
- CPU, memory, and process-count limits are set for the service.
- BLAS/OpenMP threads are capped to reduce accidental oversubscription.

Important security boundary:
`.pkl`/`.joblib` model artifacts can execute arbitrary Python during deserialization.
This compose profile provides defense-in-depth and resource isolation, but it is **not a hostile-code sandbox**. For public multi-tenant deployments, put model execution in a stronger dedicated sandbox/container/VM with network isolation and a separate security boundary.

Run from the repository root:

```bash
docker compose -f deploy/docker-compose.prod.yml up --build
```

API health:
`http://localhost:8000/health`

Web:
`http://localhost:3000`
