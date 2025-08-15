#!/usr/bin/env python
"""
Simple Strands Agent using built-in AWS Bedrock support with Braintrust OpenTelemetry
"""
import os
from dotenv import load_dotenv
from strands import Agent
from strands.models import BedrockModel
from strands import tool

load_dotenv()

# Define some tools for the agent
@tool(description="Get the current weather for a location")
def get_weather(location: str) -> str:
    """Get the weather for a given location (mock implementation)"""
    return f"The weather in {location} is sunny and 72°F with a light breeze."

@tool(description="Calculate mathematical expressions")
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely"""
    try:
        # Only allow basic math operations
        allowed = set("0123456789+-*/()., ")
        if all(c in allowed for c in expression):
            result = eval(expression, {"__builtins__": {}}, {})
            return f"The result is: {result}"
        else:
            return "Invalid expression. Only basic math operations are allowed."
    except Exception as e:
        return f"Error: {str(e)}"

def setup_braintrust_telemetry():
    """Configure OpenTelemetry for Braintrust"""
    from braintrust.otel import BraintrustSpanProcessor #from the braintrust[otel] package
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from strands.telemetry import StrandsTelemetry

    # Make sure to set the: 
    # BRAINTRUST_API_KEY=<your Braintrust API key>
    # BRAINTRUST_PARENT=project_name:<your project name>
    # BRAINTRUST_API_URL=https://api.braintrust.dev
    # in an environment file
    
    # Configure the global OTel tracer provider
    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    
    # Send spans to Braintrust.
    provider.add_span_processor(BraintrustSpanProcessor())
    telemetry = StrandsTelemetry(provider)
    
    return telemetry

async def main():
    
    # Setup telemetry
    telemetry = setup_braintrust_telemetry()
    
    # Get model ID from environment or use default
    model_id = os.getenv('BEDROCK_MODEL_ID')
    
    # Create Bedrock model
    bedrock_model = BedrockModel(
        model_id=model_id,
        region=os.getenv('AWS_REGION', 'us-east-1')
    )
    
    # Create the agent
    agent = Agent(
        model=bedrock_model,
        name="StrandsBedrockAgent",
        description="A helpful AI assistant powered by AWS Bedrock",
        system_prompt="""You are a friendly and helpful AI assistant built with the Strands SDK 
        and powered by AWS Bedrock. You have access to tools for weather information and calculations.
        Be concise and informative in your responses.""",
        tools=[get_weather, calculate],
        trace_attributes={
            "agent.type": "strands",
            "llm.provider": "aws_bedrock",
            "llm.model": model_id
        }
    )

if __name__ == "__main__":
    import asyncio
        # Test conversations
    test_inputs = [
        "What's the weather in Paris?",
        "Calculate 42 * 17 + 3",
    ]
    asyncio.run(main())