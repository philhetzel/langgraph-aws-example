# Strands-style Agent with Braintrust OpenTelemetry Integration

This example demonstrates a Strands SDK-style agent using AWS Bedrock with OpenTelemetry tracing configured for Braintrust.dev.

## Features

- **Strands Pattern**: Implements the Think -> Act -> Observe agent pattern
- **AWS Bedrock Integration**: Uses Claude models via AWS Bedrock
- **OpenTelemetry Tracing**: Full distributed tracing support
- **Braintrust Integration**: Sends traces and metrics to Braintrust.dev
- **Automatic Instrumentation**: Instruments AWS SDK calls automatically

## Setup

1. **Install Dependencies**:
```bash
uv sync
```

2. **Configure Environment Variables**:
Create a `.env` file with:
```env
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0

# Braintrust Configuration
BRAINTRUST_API_KEY=your_braintrust_api_key
# Get your project ID from Braintrust dashboard (it's a UUID)
BRAINTRUST_PROJECT_ID=your_project_uuid
```

3. **Get Your Braintrust Project ID**:
   - Go to your Braintrust dashboard
   - Navigate to your project settings
   - Click "Copy Project ID" to get the UUID
   - Use this UUID (not the project name) in `BRAINTRUST_PROJECT_ID`

## Running the Agent

```bash
uv run python strands_agent.py
```

## Architecture

The agent follows the Strands pattern:

1. **Think**: Analyzes the current state and plans the response
2. **Act**: Executes the plan using AWS Bedrock
3. **Observe**: Logs the results to Braintrust

Each step is traced with OpenTelemetry, providing:
- Span creation for each phase
- Metrics collection (cycles, tool calls, latency)
- Automatic AWS SDK instrumentation
- Error tracking and exception recording

## Telemetry Configuration

The `StrandsTelemetry` class configures:
- OTLP HTTP exporters for traces and metrics
- Braintrust-specific headers (Authorization and x-bt-parent)
- Service name and version resource attributes
- Automatic boto3 instrumentation

## Viewing Traces

After running the agent:
1. Go to your Braintrust dashboard
2. Navigate to your project
3. View the traces in the Logs section
4. Analyze metrics and performance data

## Troubleshooting

If you see "Missing read access to project_log" errors:
- Ensure you're using the correct project UUID (not the name)
- Verify your API key has access to the project
- Check that the project exists in your Braintrust account

If traces don't appear:
- Ensure you're sending at least one root span
- Check that the BRAINTRUST_API_KEY is valid
- Verify the endpoint is reachable from your network