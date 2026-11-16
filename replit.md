# DriftGuard XR

A controlled synthetic visual-inertial pose-estimation research workbench.

## Run & Operate

- `artifacts/api-server: API Server` runs the **Python FastAPI** application using
  `pnpm --filter @workspace/api-server run dev` → `python -m backend.api`.
- `artifacts/driftguard: web` runs the React frontend at `/`.
- `python -m pytest backend/tests -q` runs numerical tests.
- `pnpm run typecheck` checks TypeScript.
- `pnpm --filter @workspace/api-spec run codegen` regenerates API types/hooks.
- `DATABASE_URL` is required for durable structured experiment records.
- Python dependencies are in `pyproject.toml` / `uv.lock`; install with `uv sync --frozen`.

## Product and constraints

This is a real numerical synthetic benchmark, not general-purpose SLAM or real
headset integration. The user explicitly requires actual corrupted sensor
measurements, ESKF/PnP algorithms, honest uncertainty and computed metrics.
Never replace these with frontend-generated trajectories or hardcoded results.
No paid APIs, cloud AI or external LLM dependencies.

## Architecture

- `backend/simulation`: seeded 30 Hz camera, 200 Hz IMU, indoor landmarks.
- `backend/estimation`: PnP, right-error 15-state ESKF, health/recovery.
- `backend/goi`: observable-subspace GOI-inspired influence and keyframes.
- `backend/benchmarking`: same-input estimators, isolated ablations, metrics.
- `backend/api.py`: FastAPI endpoints; old Express scaffold is unused.
- `backend/repository.py`: PostgreSQL JSONB numerical records.
- `artifacts/driftguard`: six-route React / R3F / Recharts workbench.
- See `README.md` for equations, units, API, limitations and reproduction.

## Non-obvious rules

- Estimators accept `Measurements`, never `GroundTruth`. First-frame PnP and a
  zero-velocity prior initialize all methods; truth is evaluation-only.
- Raw GOI is an influence diagnostic, not an outlier classifier. Robust residual
  testing handles outliers. Do not reject a feature because it has high GOI.
- Simulations serialize because OpenCV RANSAC's RNG is global.
- Quaternions are xyzw; world/camera axes are OpenCV x-right/y-down/z-forward.
- Keep the three isolation ablations: diagnostics alone must preserve the pose,
  geometry covariance and keyframe selection must also be compared separately.
- Histories are immutable; PATCH computes a new experiment without overwriting.
- Do not remove uncertainty/failure disclosures to improve apparent performance.
