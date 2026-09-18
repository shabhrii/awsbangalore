# Zero-Egress Edge-to-Cloud Drift Detection

> **AWS Hackathon Project** — A production-grade, event-driven hybrid MLOps pipeline that detects AI model drift locally at the edge using statistical methods, and triggers automated retraining on AWS with **zero raw data egress**.

## Architecture

```
[Edge Device]
   |-- Local Inference Stream
   |-- Rolling Window Buffer (N=500 records)
   |-- Tier 1: PSI (Population Stability Index)
   |-- Tier 2: Kolmogorov-Smirnov Test
   |
   |--( drift detected? )--> Compressed JSON Payload (<2KB)
                                        |
                              [Amazon API Gateway]
                                        |
                               [AWS Lambda Ingest]
                                  |          |
                             [Amazon S3]  [EventBridge]
                                              |
                                    [AWS Step Functions]
                                              |
                              [SageMaker Retraining Pipeline]
                                    +  [Amazon SNS Alert]
```

## Project Structure

```
awsbangalore/
├── edge_daemon/
│   ├── baseline_generator.py   # Trains baseline, exports baseline_profile.json
│   ├── daemon.py               # PSI + KS-Test drift engine + simulation loop
│   └── requirements.txt
├── dashboard/
│   ├── app.py                  # Streamlit demo dashboard
│   └── requirements.txt
├── infrastructure/
│   ├── app.py                  # CDK App entry point
│   ├── drift_stack.py          # CDK Stack (API GW, Lambda, EventBridge, Step Functions, SNS)
│   ├── cdk.json                # CDK project configuration
│   ├── requirements.txt
│   └── lambdas/
│       └── ingest.py           # Lambda ingestion handler
├── tests/
│   └── test_daemon.py          # Unit tests for PSI + KS logic
└── README.md
```

## Setup & Run

### 1. Edge Daemon (Local)

```bash
cd edge_daemon
pip install -r requirements.txt

# Generate baseline profile
python baseline_generator.py

# Run simulation (clean data then inject drift)
python daemon.py
```

### 2. Streamlit Dashboard

```bash
cd dashboard
pip install -r requirements.txt

# First generate baseline in edge_daemon/
streamlit run app.py
```

### 3. AWS Infrastructure Deployment

```bash
cd infrastructure
pip install -r requirements.txt
npm install -g aws-cdk

# Bootstrap your AWS account (once per account/region)
cdk bootstrap

# Deploy the stack
cdk deploy
```

> After deploying, copy the `API Gateway URL` from the CDK output and update `API_GATEWAY_URL` in `edge_daemon/daemon.py`.

### 4. Run Unit Tests

```bash
pip install pytest numpy scipy scikit-learn
pytest tests/ -v
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| PSI threshold = 0.2 | Industry standard for "significant" distribution shift |
| KS p-value threshold = 0.05 | Standard statistical significance level |
| Payload < 2KB | Only distributional summaries; zero raw rows or PII sent |
| Mock SageMaker in Step Functions | Fast hackathon demo; replace `Pass` state with real SageMaker task for production |
