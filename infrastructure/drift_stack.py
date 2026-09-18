"""
infrastructure/drift_stack.py
------------------------------
AWS CDK v2 Stack for Zero-Egress Drift Detection.

Resources provisioned:
  - S3 Bucket            : Persists drift telemetry JSON payloads
  - API Gateway HTTP API : Receives <2KB compressed drift payloads from edge
  - Lambda (Ingest)      : Validates payload, writes to S3, emits EventBridge event
  - EventBridge Bus      : Custom bus for drift.edge.engine events
  - Step Functions       : Orchestrates mock retraining + SNS notification
  - SNS Topic            : Real-time alerts to DevRel/Ops
  - CloudWatch           : Custom metrics dashboard
"""

from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    CfnOutput,
    aws_s3 as s3,
    aws_apigateway as apigw,
    aws_lambda as _lambda,
    aws_events as events,
    aws_events_targets as targets,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_sns as sns,
    aws_sns_subscriptions as subscriptions,
    aws_cloudwatch as cw,
    aws_logs as logs,
)
from constructs import Construct


class DriftStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── 1. S3 Bucket — Drift Telemetry Payloads ─────────────────────────
        payload_bucket = s3.Bucket(
            self,
            "DriftPayloadBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            versioned=False,
            lifecycle_rules=[
                s3.LifecycleRule(
                    expiration=Duration.days(30),
                    enabled=True,
                )
            ],
        )

        # ── 2. EventBridge Custom Bus ────────────────────────────────────────
        drift_bus = events.EventBus(
            self,
            "DriftEventBus",
            event_bus_name="drift-edge-engine-bus",
        )

        # ── 3. SNS Alert Topic ───────────────────────────────────────────────
        alert_topic = sns.Topic(
            self,
            "DriftAlertTopic",
            display_name="Model Drift Alerts",
            topic_name="drift-alerts",
        )
        # Uncomment and set your email for real alerts:
        # alert_topic.add_subscription(subscriptions.EmailSubscription("devops@example.com"))

        # ── 4. Ingestion Lambda ──────────────────────────────────────────────
        ingest_log_group = logs.LogGroup(
            self,
            "IngestLambdaLogs",
            log_group_name="/aws/lambda/DriftIngestHandler",
            removal_policy=RemovalPolicy.DESTROY,
            retention=logs.RetentionDays.ONE_WEEK,
        )

        ingest_lambda = _lambda.Function(
            self,
            "IngestLambda",
            function_name="DriftIngestHandler",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="ingest.handler",
            code=_lambda.Code.from_asset("lambdas"),
            timeout=Duration.seconds(15),
            memory_size=256,
            environment={
                "BUCKET_NAME": payload_bucket.bucket_name,
                "EVENT_BUS_NAME": drift_bus.event_bus_name,
            },
            log_group=ingest_log_group,
        )

        payload_bucket.grant_write(ingest_lambda)
        drift_bus.grant_put_events_to(ingest_lambda)

        # ── 5. API Gateway (REST) ────────────────────────────────────────────
        api = apigw.LambdaRestApi(
            self,
            "DriftIngestApi",
            handler=ingest_lambda,
            proxy=False,
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                throttling_rate_limit=100,
                throttling_burst_limit=50,
            ),
            rest_api_name="DriftIngestionAPI",
            description="Receives compressed drift payloads from edge nodes",
        )

        # POST /v1/telemetry/drift-event
        v1 = api.root.add_resource("v1")
        telemetry = v1.add_resource("telemetry")
        drift_event = telemetry.add_resource("drift-event")
        drift_event.add_method(
            "POST",
            method_responses=[
                apigw.MethodResponse(status_code="200"),
                apigw.MethodResponse(status_code="400"),
                apigw.MethodResponse(status_code="500"),
            ],
        )

        # ── 6. Step Functions State Machine ──────────────────────────────────
        # For the hackathon: Pass state mocks SageMaker pipeline trigger.
        # In production, replace with tasks.SageMakerCreateTrainingJob.
        retrain_mock = sfn.Pass(
            self,
            "TriggerSageMakerRetraining",
            comment="Mock: In production, trigger SageMaker Pipelines here",
            result=sfn.Result.from_object(
                {"status": "Retraining Triggered", "pipeline": "drift-retrain-v1"}
            ),
            result_path="$.retraining",
        )

        sns_notify = tasks.SnsPublish(
            self,
            "NotifyDevOps",
            topic=alert_topic,
            message=sfn.TaskInput.from_json_path_at("States.Format('🚨 Model Drift Detected on Node {} — Retraining Pipeline Triggered', $.node_id)"),
            subject="Model Drift Alert",
            result_path="$.notification",
        )

        # CDK v2 — use definition_body instead of deprecated `definition`
        definition_body = sfn.DefinitionBody.from_chainable(
            retrain_mock.next(sns_notify)
        )

        state_machine = sfn.StateMachine(
            self,
            "DriftOrchestrator",
            state_machine_name="DriftRetrainingOrchestrator",
            definition_body=definition_body,
            timeout=Duration.minutes(10),
            tracing_enabled=True,
        )

        # ── 7. EventBridge Rule → Step Functions ────────────────────────────
        events.Rule(
            self,
            "DriftCriticalRule",
            event_bus=drift_bus,
            description="Routes CRITICAL drift events to the retraining orchestrator",
            event_pattern=events.EventPattern(
                source=["drift.edge.engine"],
                detail_type=["ModelDriftDetected"],
                detail={
                    "drift_severity": ["CRITICAL"],
                },
            ),
            targets=[targets.SfnStateMachine(state_machine)],
        )

        # WARNING events → SNS directly (no retraining needed, just alert)
        events.Rule(
            self,
            "DriftWarningRule",
            event_bus=drift_bus,
            description="Routes WARNING drift events directly to SNS",
            event_pattern=events.EventPattern(
                source=["drift.edge.engine"],
                detail_type=["ModelDriftDetected"],
                detail={
                    "drift_severity": ["WARNING"],
                },
            ),
            targets=[targets.SnsTopic(alert_topic)],
        )

        # ── 8. CloudWatch Dashboard ──────────────────────────────────────────
        dashboard = cw.Dashboard(
            self,
            "DriftDashboard",
            dashboard_name="ZeroEgressDriftMonitor",
        )

        dashboard.add_widgets(
            cw.SingleValueWidget(
                title="Lambda Invocations",
                metrics=[
                    ingest_lambda.metric_invocations(
                        statistic="Sum",
                        period=Duration.minutes(5),
                    )
                ],
                width=6,
                height=3,
            ),
            cw.SingleValueWidget(
                title="Lambda Errors",
                metrics=[
                    ingest_lambda.metric_errors(
                        statistic="Sum",
                        period=Duration.minutes(5),
                    )
                ],
                width=6,
                height=3,
            ),
            cw.GraphWidget(
                title="Step Functions Executions",
                left=[
                    state_machine.metric_started(period=Duration.minutes(5)),
                    state_machine.metric_succeeded(period=Duration.minutes(5)),
                    state_machine.metric_failed(period=Duration.minutes(5)),
                ],
                width=12,
                height=6,
            ),
        )

        # ── 9. CFN Outputs ───────────────────────────────────────────────────
        CfnOutput(self, "ApiEndpointUrl", value=f"{api.url}v1/telemetry/drift-event",
                  description="Paste this into edge_daemon/daemon.py → API_GATEWAY_URL")
        CfnOutput(self, "PayloadBucketName", value=payload_bucket.bucket_name)
        CfnOutput(self, "AlertTopicArn", value=alert_topic.topic_arn)
        CfnOutput(self, "StateMachineArn", value=state_machine.state_machine_arn)
