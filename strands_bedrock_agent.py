"""
Hello World Strands Agent with AWS Bedrock and Braintrust OpenTelemetry
"""
import os
import json
from typing import Optional, Any
from dotenv import load_dotenv
import boto3
from botocore.config import Config

from strands import Agent
from strands.models import Model
from strands.telemetry import StrandsTelemetry
from strands import tool

load_dotenv()

class BedrockModel(Model):
    """Custom AWS Bedrock model for Strands SDK"""
    
    def __init__(self, model_id: str = None, region: str = 'us-east-1'):
        super().__init__()
        self.bedrock_runtime = boto3.client(
            'bedrock-runtime',
            region_name=region,
            config=Config(read_timeout=300, retries={'max_attempts': 3}),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # Handle model selection - use models that work with on-demand throughput
        if model_id and 'sonnet-4' in model_id:
            # Claude Sonnet 4 requires inference profile, fall back to Sonnet 3
            self.model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
            print(f"Using fallback model: {self.model_id}")
        elif model_id:
            self.model_id = model_id
        else:
            self.model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')
        
        print(f"Initialized Bedrock model: {self.model_id}")
        self.config = {"model_id": self.model_id, "region": region}
    
    def __call__(self, messages, tools=None, **kwargs):
        """Run inference using AWS Bedrock"""
        # Convert messages to Bedrock format
        bedrock_messages = []
        system_prompt = None
        
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                if msg.role == "system":
                    system_prompt = msg.content
                else:
                    bedrock_messages.append({
                        "role": msg.role,
                        "content": msg.content if isinstance(msg.content, str) else str(msg.content)
                    })
            else:
                # Handle dict messages
                if msg.get('role') == 'system':
                    system_prompt = msg.get('content', '')
                else:
                    bedrock_messages.append({
                        "role": msg.get('role', 'user'),
                        "content": msg.get('content', '')
                    })
        
        # Prepare the request body for Claude models
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": bedrock_messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        # Add tools if provided
        if tools:
            tool_definitions = []
            for t in tools:
                if hasattr(t, '__tool_metadata__'):
                    # Extract tool metadata
                    tool_def = {
                        "name": t.__tool_metadata__.get('name', t.__name__),
                        "description": t.__tool_metadata__.get('description', ''),
                        "input_schema": {
                            "type": "object",
                            "properties": t.__tool_metadata__.get('parameters', {}),
                            "required": t.__tool_metadata__.get('required', [])
                        }
                    }
                    tool_definitions.append(tool_def)
            
            if tool_definitions:
                body["tools"] = tool_definitions
        
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                contentType='application/json',
                accept='application/json',
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract the response content
            content = response_body.get('content', [])
            if content and len(content) > 0:
                # Check if there's a tool use request
                for item in content:
                    if item.get('type') == 'tool_use':
                        # Return tool call information
                        return {
                            'content': '',
                            'tool_calls': [{
                                'name': item.get('name'),
                                'arguments': item.get('input', {})
                            }]
                        }
                
                # Regular text response
                text = content[0].get('text', '')
                return {'content': text}
            
            return {'content': ''}
            
        except Exception as e:
            print(f"Error calling Bedrock: {e}")
            return {'content': f"Error: {str(e)}"}
    
    def get_config(self):
        """Return model configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update model configuration"""
        self.config.update(kwargs)
    
    def stream(self, messages, tools=None, system_prompt=None, **kwargs):
        """Stream responses (not implemented for this example)"""
        # For simplicity, just return the full response
        if system_prompt:
            # Add system prompt to messages if provided
            messages = [{"role": "system", "content": system_prompt}] + list(messages)
        result = self.__call__(messages, tools, **kwargs)
        yield result
    
    def structured_output(self, messages, schema, **kwargs):
        """Get structured output (not implemented for this example)"""
        # For simplicity, just return regular output
        return self.__call__(messages, **kwargs)

# Define some simple tools for the agent
@tool(description="Get the current weather for a location")
def get_weather(location: str) -> str:
    """Get the weather for a given location (mock implementation)"""
    return f"The weather in {location} is sunny and 72°F"

@tool(description="Perform basic arithmetic calculations")
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

def create_strands_agent_with_telemetry():
    """Create a Strands agent with Braintrust telemetry"""
    
    # Initialize telemetry
    telemetry = StrandsTelemetry()
    
    # Configure OTLP for Braintrust
    braintrust_api_key = os.getenv("BRAINTRUST_API_KEY")
    braintrust_project_id = os.getenv("BRAINTRUST_PROJECT_ID", "LangGraphBedrock")
    
    if braintrust_api_key:
        # Set environment variables for OTLP
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://api.braintrust.dev/otel"
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Bearer {braintrust_api_key},x-bt-parent=project_name:{braintrust_project_id}"
        
        # Setup OTLP exporter
        telemetry.setup_otlp_exporter()
        print(f"✅ Configured Braintrust telemetry for project: {braintrust_project_id}")
    else:
        # Use console exporter if no Braintrust key
        telemetry.setup_console_exporter()
        print("Using console telemetry (no Braintrust API key found)")
    
    # Create the Bedrock model
    bedrock_model = BedrockModel()
    
    # Create the agent
    agent = Agent(
        model=bedrock_model,
        name="StrandsBedrockAgent",
        description="A Strands agent powered by AWS Bedrock with OpenTelemetry tracing",
        system_prompt="""You are a helpful AI assistant built with the Strands SDK and powered by AWS Bedrock. 
        You have access to tools for weather information and calculations.
        Be concise and friendly in your responses.""",
        tools=[get_weather, calculate],
        trace_attributes={
            "agent.type": "strands",
            "llm.provider": "aws_bedrock",
            "project": braintrust_project_id or "default"
        }
    )
    
    return agent

async def main():
    """Run the Strands agent with test inputs"""
    print("=" * 60)
    print("🚀 Strands Agent with AWS Bedrock and Braintrust")
    print("=" * 60)
    
    # Create the agent
    agent = create_strands_agent_with_telemetry()
    
    # Test conversations
    test_conversations = [
        "Hello! Can you introduce yourself?",
        "What's the weather like in San Francisco?",
        "Can you calculate 42 * 17 for me?",
        "Tell me a short joke about programming.",
    ]
    
    for user_input in test_conversations:
        print(f"\n👤 User: {user_input}")
        print("-" * 40)
        
        # Run the agent
        response = await agent.invoke_async(user_input)
        
        # Display response
        print(f"🤖 Agent: {response}")
    
    print("\n" + "=" * 60)
    print("✅ Strands agent demo completed!")
    print("📊 Check your Braintrust dashboard for telemetry data")
    print("=" * 60)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())