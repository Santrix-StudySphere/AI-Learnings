## What is an AI Agent?
You've probably used an LLM like Gemini before, where you give it a prompt and it gives you a text response.

Prompt -> LLM -> Text

An AI Agent takes this one step further. An agent can think, take actions, and observe the results of those actions to give you a better answer.

Prompt -> Agent -> Thought -> Action -> Observation -> Final Answer

## Define your agent:
```
root_agent = Agent(
    name="helpful_assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that can answer general questions.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)
```
## Run your agent:
 To do this, you need a Runner, which is the central component within ADK that acts as the orchestrator. 
 It manages the conversation, sends our messages to the agent, and handles its responses.
```
runner = InMemoryRunner(agent=root_agent)

print("✅ Runner created.")
```

## Try the ADK Web Interface:

Run the command below to generate a sample-agent folder that contains all the necessary files,
including agent.py for your code, an .env file with your API key pre-configured, and an __init__.py file:
```
!adk create sample-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY
```
```
url_prefix = get_adk_proxy_url()
```
```
!adk web --url_prefix {url_prefix}  
```

## Now you can call the .run_debug() method to send our prompt and get an answer.
```
response = await runner.run_debug(
    "What is Agent Development Kit from Google? What languages is the SDK available in?"
)
```
