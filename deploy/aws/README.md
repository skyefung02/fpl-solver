# Deploying the market collector

Polls `bootstrap-static` every 5 minutes and writes the raw payload to S3. Replaces a GitHub
Actions cron that delivered ~72% of its scheduled runs.

## 0. Prerequisites

`aws` and `sam` must both be on PATH:

```bash
aws --version    # aws-cli/2.x
sam --version    # SAM CLI 1.x
```

AWS CLI v2 is installed from AWS's own package, not Homebrew — `brew install awscli` fails on
this machine because `/usr/local/bin` has a Python 3.8 framework shadowing brew's Python.

## 1. IAM user and access keys (AWS console)

Console → search **IAM** → **Users** → **Create user**

- User name: `fpl-deployer`
- Leave **"Provide user access to the AWS Management Console"** UNCHECKED — this is CLI-only.
- Permissions → **Attach policies directly** → `AdministratorAccess`

  Deploying this stack touches CloudFormation, Lambda, S3, IAM, EventBridge Scheduler, SQS, SNS
  and CloudWatch. Scoping that precisely is fiddly; admin on a personal account is the pragmatic
  choice. Scope it down later if you want.

Then open the user → **Security credentials** tab → **Access keys** → **Create access key**
→ use case **Command Line Interface (CLI)** → acknowledge → **Create access key**.

The secret is shown **once**. Copy both values or download the CSV.

## 2. Configure the CLI

```bash
aws configure
#   AWS Access Key ID     : <from step 1>
#   AWS Secret Access Key : <from step 1>
#   Default region name   : ap-southeast-2
#   Default output format : json

aws sts get-caller-identity     # must return an ARN before going further
```

## 3. Deploy

`sam build` needs a `python3.13` on PATH matching the Lambda runtime:

```bash
cd deploy/aws
PATH="/opt/miniconda3/envs/open-fpl-solver/bin:$PATH" sam build
sam deploy --guided
```

Answers:

| Prompt | Answer |
|---|---|
| Stack Name | `fpl-market-snapshot` |
| AWS Region | `ap-southeast-2` |
| Parameter AlertEmail | your email address |
| Parameter ScheduleExpression | accept default `rate(5 minutes)` |
| Parameter MinRunsPerHour | accept default `6` |
| Confirm changes before deploy | `y` — shows the changeset before anything is created |
| Allow SAM CLI IAM role creation | `Y` |
| Disable rollback | `N` |
| Save arguments to configuration file | `Y` (writes `samconfig.toml`, so later deploys are just `sam deploy`) |

Note the `BucketName` in the stack outputs.

## 4. Confirm the alarm email

Check your inbox for **"AWS Notification - Subscription Confirmation"** and click
**Confirm subscription**. Until you do, every alarm fires into a void.

## 5. Verify it is collecting

Set the console region selector (top right) to **Sydney ap-southeast-2** or you will see nothing.

```bash
aws s3 ls s3://<bucket>/raw/ --recursive | tail -5    # new object every ~5 min
```

- **S3** → your bucket → `raw/` — objects appear as `2026/08/28/HHMMSSffffff-aNNN.json.gz`
- **Lambda** → Functions → `...CollectorFunction...` → **Monitor** tab — invocations, errors, duration
- **EventBridge** → left nav **Scheduler** → **Schedules** — the 5-minute schedule, state Enabled
- **CloudWatch** → **Alarms** → three alarms, all should settle to OK

## 6. Test the absence alarm

This is the alarm the whole migration exists for, so prove it works:

1. **EventBridge** → **Scheduler** → **Schedules** → select the schedule → **Actions** → **Disable**
2. Wait. The alarm uses a 3600s period with 1 evaluation period, so allow **up to ~2 hours**
   depending on where you land in the hour.
3. You should get an email and see the alarm go to **In alarm** in CloudWatch.
4. **Enable** the schedule again. `OKActions` is set, so you also get the recovery email.

An untested alarm is not an alarm.

## 7. Set a billing guard

Console → **Billing and Cost Management** → **Budgets** → **Create budget** → template
**Monthly cost budget** → $5 → your email.

Expected real cost is a few cents a month: 8,640 invocations against a 1,000,000 free tier, and
~10.6GB of S3 over nine months. The budget is there to catch a mistake, not the normal bill.

## Pulling data down for training

```bash
aws s3 sync s3://<bucket>/raw ./data/market_raw
python run/build_market_table.py --raw data/market_raw \
    --legacy /path/to/data-branch/snapshots -o data/market_table.parquet
```

`--legacy` reads the GitHub-era CSV parts, so the seam between the two collectors does not reach
the model. Drop it once the legacy window is no longer relevant.
