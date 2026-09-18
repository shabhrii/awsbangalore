import aws_cdk as cdk
from drift_stack import DriftStack

app = cdk.App()

DriftStack(
    app,
    "ZeroEgressDriftStack",
    description="Zero-Egress Edge-to-Cloud Drift Detection — AWS Hackathon",
    env=cdk.Environment(
        # Reads AWS_DEFAULT_REGION / AWS_ACCOUNT_ID from environment if set,
        # otherwise CDK resolves from the active CLI profile at synth time.
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region"),
    ),
)

cdk.Tags.of(app).add("Project", "ZeroEgressDriftDetection")
cdk.Tags.of(app).add("Environment", "hackathon")

app.synth()
