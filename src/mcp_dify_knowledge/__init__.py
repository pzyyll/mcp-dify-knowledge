import httpx
import os
import asyncio

from mcp.server import Server, stdio
from mcp.types import Tool, TextContent

from typing import Annotated
from pydantic import BaseModel, Field

mcp_server = Server("dify-knowledge")


DIFY_API_URL = os.getenv("DIFY_API_URL", "your_dify_api_url_here")
DIFY_API_KEY = os.getenv("DIFY_API_KEY", "your_api_key_here")

headers = {"Authorization": f"Bearer {DIFY_API_KEY}"}


class ListKnowledgeParams(BaseModel):
    keyword: Annotated[
        str,
        Field(default="", description="Keyword to filter knowledge bases."),
    ]


class QueryKnowledgeParams(BaseModel):
    id: Annotated[
        str,
        Field(description="ID of the knowledge base to query."),
    ]
    query: Annotated[
        str,
        Field(description="Query to search in the knowledge base."),
    ]


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    return [
        Tool(
            name="list_knowledge",
            description="List all knowledge bases.",
            inputSchema=ListKnowledgeParams.model_json_schema(),
        ),
        Tool(
            name="query_knowledge",
            description="Query a specific knowledge base.",
            inputSchema=QueryKnowledgeParams.model_json_schema(),
        ),
    ]


@mcp_server.call_tool()
async def call_tool(tool_name: str, params: dict) -> list[TextContent]:
    """Call a tool with the given name and parameters."""
    if tool_name == "list_knowledge":
        return await list_knowledge(ListKnowledgeParams(**params))
    elif tool_name == "query_knowledge":
        return await query_knowledge(QueryKnowledgeParams(**params))

    raise ValueError(f"Tool '{tool_name}' not found.")


async def list_knowledge(params: ListKnowledgeParams) -> list[TextContent]:
    """List knowledge bases from the Dify API."""
    async with httpx.AsyncClient() as client:
        h = headers.copy()
        h["Content-Type"] = "application/json"

        response = await client.get(
            f"{DIFY_API_URL}/datasets",
            headers=h,
            params=params.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        return [TextContent(type="text", text=response.text)]


async def query_knowledge(params: QueryKnowledgeParams) -> list[TextContent]:
    """Query a specific knowledge base through the Dify API."""

    transport = httpx.AsyncHTTPTransport(retries=3)
    async with httpx.AsyncClient(timeout=60, transport=transport) as client:
        try:
            json_payload = params.model_dump(exclude_none=True, exclude={"id"})
            url = f"{DIFY_API_URL}/datasets/{params.id}/retrieve"

            request_headers = headers.copy()
            request_headers["Content-Type"] = "application/json"

            response = await client.post(
                url,
                headers=request_headers,
                json=json_payload
            )

            response.raise_for_status()
            return [TextContent(type="text", text=response.text)]

        except httpx.HTTPStatusError as e_http:
            response_text_summary = (
                e_http.response.text[:200]
                if hasattr(e_http.response, "text")
                else "N/A"
            )
            error_message = f"Dify API HTTP Error: {e_http.response.status_code}. Response: {response_text_summary}"
            return [TextContent(type="text", text=error_message)]

        except httpx.RequestError as e_req:
            error_message = f"Dify API Request Error: {str(e_req)}"
            return [TextContent(type="text", text=error_message)]

        except Exception as e:
            error_message = f"Querying Dify API encountered an unexpected error: {str(e)}"
            return [TextContent(type="text", text=error_message)]


async def serve() -> None:
    options = mcp_server.create_initialization_options()
    async with stdio.stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream, write_stream, options, raise_exceptions=True
        )


def main() -> None:
    """Main function to run the Dify Knowledge module."""
    # print to stderr
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("Server shutting down...")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
