import os
import chainlit as cl
from typing import cast
from agents import Agent, Runner, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from agents.run import RunConfig
from dotenv import load_dotenv
from travel_agents import travel_agent

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("GEMINI_BASE_URL")
model = "gemini-2.0-flash"

@cl.on_chat_start
async def chat_start():

    client = AsyncOpenAI(api_key= gemini_api_key, base_url= base_url)

    config = RunConfig(model= OpenAIChatCompletionsModel(model=model,openai_client= client),
                       model_provider=client,
                       tracing_disabled=True)
    
    cl.user_session.set("config", config)
    cl.user_session.set("chathistory",[])
    cl.user_session.set("agent",travel_agent)
    await cl.Message("Welcome to Insia's Travel Agent👋 What do you want to know?").send()

@cl.on_message
async def main(message:cl.Message):
    msg= cl.Message(content="")
    await msg.send()

    agent:Agent = cast(Agent,cl.user_session.get("agent"))
    config = cast(RunConfig,cl.user_session.get("config"))
    history = cl.user_session.get("chathistory") or []
    history.append({"role":"user","content":message.content})

    try:
        print("\n CALLING AGENT WITH CONTEXT\n",history,"\n")
        result = await Runner.run(starting_agent= agent, input= history, run_config= config)
        msg.content = result.final_output
        await msg.update()
        cl.user_session.set("chathistory",result.to_input_list())
        print(f"User: {message.content}")
        print(f"AI Assistant: {msg.content}")
        
    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")
        