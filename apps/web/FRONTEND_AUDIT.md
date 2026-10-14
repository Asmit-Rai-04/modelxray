# Frontend audit and implementation scope

Source: GitHub `main`, commit `830e6ba` (`feat: redesign ModelXray frontend`). The supplied unpacked release is retained unchanged outside this clone. Work is isolated on `feat/premium-investigation-workspace`.

## Already working on main

- Single-model upload setup is separate from benchmark investigation and regression.
- Model/dataset artifact registration and uploaded-model comparison use real API endpoints.
- Five workspaces have distinct empty states; completed runs expose atlas details, experiments and coverage.
- Theme preference persists; identity uses a wordmark rather than the old MX square.
- Evidence nodes and paths already used one SVG coordinate space.
- Regression already uses percentage-point differences and cluster-aware backend deltas.

## Gaps identified

- Frontend used synchronous investigation calls despite persistent async jobs and cancellation support.
- No job recovery, saved-investigation navigation or artifact-selection persistence.
- Unconditional ENGINE READY label, inert Documentation action, absent comparability warnings.
- 7–11px metadata and low-contrast secondary text in both themes; incomplete focus/reduced-motion treatment.
- Atlas lacked search/severity/sorting controls; effect intervals lacked charts and exact-value table alternatives.
- Monolithic page mixed request orchestration and all analytical workspaces.
- Next.js 15.5.4 dependency tree reported one critical and three high npm vulnerabilities.

## Implementation

Preserve business logic and API contracts. Split the shell, setup, Overview, Atlas, Experiments, Coverage, Regression, shared controls and charts; centralize API types/formatting; isolate persistent job lifecycle in a hook. Refine the visual system without adding production dependencies. Add browser component/interaction checks and real API end-to-end paths using backend-generated fixtures. Upgrade within the Next.js 15.5 and React 19.1 release lines and test explicit transitive dependency overrides.

## Backend contracts used

| Capability | Endpoint / behavior |
| --- | --- |
| Health | `GET /health` on mount and every 30s |
| Registration | `POST /api/v1/assets/model`, `/assets/dataset` |
| Async investigation | `POST /api/v1/jobs/demo/investigate`, `/jobs/investigate/uploaded` |
| Progress/recovery | `GET /api/v1/jobs/{id}`, persisted identifier scoped to API URL |
| Cancellation | `POST /api/v1/jobs/{id}/cancel`, cooperative checkpoints |
| History | `GET /api/v1/investigations?limit=100` |
| Evidence retrieval | `GET /api/v1/investigations/{id}` |
| Regression | Synchronous `POST /api/v1/demo/compare`, `/regression/uploaded` |

Uploaded jobs report coarse worker milestones, not per-experiment progress. The UI displays exactly those server values. Regression has no async progress/cancellation API. No asset-list endpoint exists; artifact identifiers are remembered locally and backend errors handle stale references. No saved-comparison retrieval contract was found; investigations recover, comparison responses do not. No backend statistical or ML code is changed.
