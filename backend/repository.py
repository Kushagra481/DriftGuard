"""Durable numerical experiment records in the project's PostgreSQL database.

The JSONB fields are structured sensor/estimation data, not uploaded files.
Schema is managed in development and by the platform's publishing migration.
"""
import os
import json
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def connection():
    return psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)


def save(result, details):
    config = result["config"]
    metric = next(m for m in result["metrics"] if m["id"] == "driftguard")
    # Reject non-finite results explicitly rather than persisting invalid JSON.
    json.dumps(result, allow_nan=False)
    json.dumps(details, allow_nan=False)
    with connection() as conn:
        conn.execute(
            "INSERT INTO driftguard_experiments "
            "(id, created_at, preset, seed, duration, ate, result, details) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (result["id"], result["createdAt"], config["preset"], config["seed"],
             config["duration"], metric["ate"], Jsonb(result), Jsonb(details)))


def history():
    with connection() as conn:
        rows = conn.execute(
            "SELECT id, created_at, preset, seed, duration, ate "
            "FROM driftguard_experiments ORDER BY created_at DESC LIMIT 30").fetchall()
    return [{"id": x["id"], "createdAt": x["created_at"].isoformat(),
             "preset": x["preset"], "seed": x["seed"], "duration": x["duration"], "ate": x["ate"]}
            for x in rows]


def get_result(experiment_id):
    with connection() as conn:
        row = conn.execute("SELECT result FROM driftguard_experiments WHERE id=%s", (experiment_id,)).fetchone()
    return row["result"] if row else None


def get_detail(experiment_id, index):
    with connection() as conn:
        row = conn.execute(
            "SELECT result->'frames'->%s AS frame, details->%s AS detail "
            "FROM driftguard_experiments WHERE id=%s", (index, index, experiment_id)).fetchone()
    if not row or row["frame"] is None:
        return None
    return {**row["detail"], "frame": row["frame"]}


def get_details(experiment_id):
    with connection() as conn:
        row = conn.execute("SELECT details FROM driftguard_experiments WHERE id=%s", (experiment_id,)).fetchone()
    return row["details"] if row else None
