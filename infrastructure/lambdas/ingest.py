"""
infrastructure/lambdas/ingest.py
---------------------------------
Lambda handler for the Drift Ingestion API.

Responsibilities:
  1. Validate the incoming drift payload JSON schema
  2. Persist the payload to S3 (keyed by node_id/timestamp)
  3. Emit a `ModelDriftDetected` event onto the custom EventBridge bus
"""

import json
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
events = boto3.client("events")

BUCKET_NAME: str = os.environ["BUCKET_NAME"]
EVENT_BUS_NAME: str = os.environ["EVENT_BUS_NAME"]

REQUIRED_FIELDS = {"node_id", "timestamp", "drift_severity", "drifted_features"}


def _cors_headers() -> dict:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Content-Type": "application/json",
    }


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": _cors_headers(),
        "body": json.dumps(body),
    }


def handler(event: dict, context) -> dict:
    """Main Lambda entry point."""
    logger.info("Received event: %s", json.dumps(event))

    # ── 1. Parse body ────────────────────────────────────────────────────────
    try:
        raw_body = event.get("body", "{}")
        body: dict = json.loads(raw_body) if isinstance(raw_body, str) else raw_body
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON body: %s", exc)
        return _response(400, {"error": "Invalid JSON payload"})

    # ── 2. Schema validation ─────────────────────────────────────────────────
    missing = REQUIRED_FIELDS - body.keys()
    if missing:
        logger.warning("Missing required fields: %s", missing)
        return _response(400, {"error": f"Missing fields: {sorted(missing)}"})

    node_id: str = str(body["node_id"])
    severity: str = str(body.get("drift_severity", "UNKNOWN"))

    # ── 3. Persist to S3 ────────────────────────────────────────────────────
    timestamp_ms = int(time.time() * 1000)
    object_key = f"{node_id}/{timestamp_ms}.json"

    try:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=object_key,
            Body=json.dumps(body).encode("utf-8"),
            ContentType="application/json",
            Metadata={
                "node_id": node_id,
                "severity": severity,
            },
        )
        logger.info("Payload persisted → s3://%s/%s", BUCKET_NAME, object_key)
    except ClientError as exc:
        logger.error("S3 write failed: %s", exc)
        return _response(500, {"error": "Failed to persist payload"})

    # ── 4. Emit EventBridge event ────────────────────────────────────────────
    try:
        events.put_events(
            Entries=[
                {
                    "Source": "drift.edge.engine",
                    "DetailType": "ModelDriftDetected",
                    "Detail": json.dumps(body),
                    "EventBusName": EVENT_BUS_NAME,
                }
            ]
        )
        logger.info("EventBridge event emitted: ModelDriftDetected (severity=%s)", severity)
    except ClientError as exc:
        # Event emission failure is non-fatal — payload is already in S3
        logger.error("EventBridge emission failed: %s", exc)

    return _response(
        200,
        {
            "message": "Drift payload ingested successfully",
            "node_id": node_id,
            "s3_key": object_key,
            "severity": severity,
        },
    )
