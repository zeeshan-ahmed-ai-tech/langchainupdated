import os
import asyncio

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Make sure your .env file contains:\n"
        "GROQ_API_KEY=your_api_key"
    )

print("GROQ API KEY LOADED")


# ============================================================
# 2. MAIN FUNCTION
# ============================================================

async def main():

    print("Starting MCP Client...")

    # --------------------------------------------------------
    # MCP SERVERS
    # --------------------------------------------------------

    client = MultiServerMCPClient(
        {
            # -------------------------
            # Math MCP Server
            # -------------------------
            "math": {
                "command": "python",
                "args": ["mathserver.py"],
                "transport": "stdio",
            },

            # -------------------------
            # Weather MCP Server
            # -------------------------
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            },
        }
    )

    # --------------------------------------------------------
    # GET TOOLS FROM MCP SERVERS
    # --------------------------------------------------------

    print("Loading MCP tools...")

    tools = await client.get_tools()

    print(f"Loaded {len(tools)} tools")

    for tool in tools:
        print(f"- {tool.name}")

    # --------------------------------------------------------
    # GROQ MODEL
    # --------------------------------------------------------

    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    # --------------------------------------------------------
    # CREATE LANGGRAPH REACT AGENT
    # --------------------------------------------------------

    agent = create_react_agent(
        model=model,
        tools=tools,
    )

    print("Agent created successfully")

    # ========================================================
    # 3. MATH TEST
    # ========================================================

    print("\n--- MATH TEST ---")

    math_response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What's (3 + 5) x 12?",
                }
            ]
        }
    )

    print(
        "Math response:",
        math_response["messages"][-1].content,
    )

    # ========================================================
    # 4. WEATHER TEST
    # ========================================================

    print("\n--- WEATHER TEST ---")

    weather_response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in California?",
                }
            ]
        }
    )

    print(
        "Weather response:",
        weather_response["messages"][-1].content,
    )


# ============================================================
# 5. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
