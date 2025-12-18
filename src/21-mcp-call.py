from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import os


load_dotenv()

amap_api_key = os.getenv("AMAP_API_KEY")


amap_mcp_client = MultiServerMCPClient({
    # 高德地图MCP Server
    "amap-amap-sse": {
        "url": f"https://mcp.amap.com/sse?key={amap_api_key}",
        "transport": "sse",
    }
})

async def get_amap_mcp_tools():
    tools = await amap_mcp_client.get_tools()
    return tools


if __name__ == "__main__":
    import asyncio
    tools = asyncio.run(get_amap_mcp_tools())
    print(tools)

