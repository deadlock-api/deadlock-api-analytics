import functools
import json
import operator
import os
import re
from typing import TypedDict, Annotated, Literal

import clickhouse_driver
import sqlglot
from fastapi import APIRouter
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from starlette.requests import Request
from starlette.responses import Response

from deadlock_analytics_api.rate_limiter import limiter
from deadlock_analytics_api.rate_limiter.models import RateLimit

router = APIRouter(prefix="/v1", tags=["V1"])


def get_ch_client() -> clickhouse_driver.Client:
    return clickhouse_driver.Client(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_PORT", 9000)),
        user=os.getenv("CLICKHOUSE_READONLY_USER", "default"),
        password=os.getenv("CLICKHOUSE_READONLY_PASSWORD", ""),
        database=os.getenv("CLICKHOUSE_DB", "default"),
    )


class AgentState(TypedDict):
    user_query: str
    sql_queries: list[str] | None
    schema_info: str
    clickhouse_result: list[dict] | None
    formatted_response: str | None
    error: str | None
    intermediate_steps: Annotated[list[BaseMessage], operator.add]
    is_valid: bool | None


def get_table_names() -> list[str]:
    with get_ch_client() as client:
        return [
            t
            for (t,) in client.execute(
                """
            SELECT name
            FROM system.tables
            WHERE database = 'default'
                and name NOT LIKE '.%'
                and name NOT LIKE 't_%'
                and name not in ('active_matches', 'finished_matches', 'match_salts', 'mmr_history', 'player_card', 'match_player_encoded_items','match_player_item_v2')
                and engine != 'MaterializedView'
            """
            )
        ]


def describe_table(table_name: str) -> str:
    def format_column(comment, name, type):
        return (
            f"<column name='{name}' type='{type}' comment='{comment}'/>"
            if comment
            else f"<column name='{name}' type='{type}'/>"
        )

    with get_ch_client() as client:
        table_description = "\n  ".join(
            format_column(comment, name, type)
            for name, type, alias, _, comment, *_ in client.execute(f"DESCRIBE TABLE {table_name}")
            if alias == "" and "." not in name
        )
    return f"<table name='{table_name}'>\n{table_description}\n</table>"


def strip_thinking(message: str) -> str:
    return message.split("</think>")[-1].strip() if "</think>" in message else message


def parse_sql_queries(message: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"<QUERY>(.*?)</QUERY>", message, re.DOTALL)]


#### Build Graph
def generate_sql_query(state: AgentState, llm) -> dict:
    schema_info = "\n\n".join(describe_table(t) for t in get_table_names())
    prompt = """
You are an expert SQL query writer for ClickHouse.
Given a user query and a database schema, your task is to generate one or more SQL queries that retrieve the data necessary to answer the user's request.
Enclose each SQL query within `<QUERY>...</QUERY>` tags.
Ensure all queries are compatible with ClickHouse syntax.

**Database Schema:**
{schema}

**User Query:**
{input}

**Output Format:**
Provide only the SQL query (or queries) enclosed in the `<QUERY>...</QUERY>` tags.
If multiple queries are needed, separate put them in separate `<QUERY>...</QUERY>` tags.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder(variable_name="messages"), ("user", prompt)]
    )
    chain = prompt_template | llm
    result = chain.invoke(
        {
            "input": state["user_query"],
            "schema": schema_info,
            "messages": state.get("intermediate_steps", []),
        }
    )
    result = strip_thinking(result.content)
    try:
        return {
            "sql_queries": parse_sql_queries(result),
            "intermediate_steps": [AIMessage(content=result)],
            "schema_info": schema_info,
        }
    except ValueError as e:
        return {"error": str(e), "intermediate_steps": [AIMessage(content=result)]}


def fix_sql(state: AgentState, llm) -> dict:
    prompt = """
You are an expert ClickHouse SQL developer.
Your task is to refine or correct provided SQL queries to accurately answer a given user query.
You will receive the user's query, existing SQL queries (if any), and any errors encountered during execution of those queries.

Your response should contain one or more valid, ClickHouse-compatible SQL queries, each enclosed within `<QUERY>...</QUERY>` tags.
Multiple queries can be provided if necessary to achieve the desired result.
If no changes are needed, simply return the original query within the tags.
If you are creating a new query, be sure it is optimized for ClickHouse.

User Query: {user_query}

Existing SQL Queries:
{sql_queries}

Errors encountered (if any):
{error}

Think carefully about the user query and the existing SQL queries to ensure your response is accurate and helpful.
After solving the issue, step by step, explain your reasoning and the changes you made to the SQL queries.
Provide the corrected or refined SQL query (or queries) within the `<QUERY>...</QUERY>` tags.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder(variable_name="messages"), ("user", prompt)]
    )
    chain = prompt_template | llm

    result = chain.invoke(
        {
            "messages": state.get("intermediate_steps", []),
            "user_query": state["user_query"],
            "sql_queries": "\n\n".join(state.get("sql_queries", [])),
            "error": state["error"],
        }
    )
    result = strip_thinking(result.content)
    try:
        fixed_sql_queries = parse_sql_queries(result)
        if fixed_sql_queries == state.get("sql_queries"):
            return {
                "error": "Unable to fix SQL query.",
                "intermediate_steps": [AIMessage(content=result)],
            }
        return {
            "sql_queries": fixed_sql_queries,
            "intermediate_steps": [AIMessage(content=result)],
            "error": None,
        }
    except ValueError as e:
        return {"error": str(e)}


def execute_sql(state: AgentState) -> dict:
    try:
        with get_ch_client() as client:
            results = []
            for sql_query in state.get("sql_queries", []):
                try:
                    result, keys = client.execute(sql_query, with_column_types=True)
                except Exception:
                    sql_query = sqlglot.transpile(sql_query, read="postgres", write="clickhouse")[0]
                    result, keys = client.execute(sql_query, with_column_types=True)
                keys = [k for k, _ in keys]
                results.append([dict(zip(keys, row)) for row in result])
            return {
                "clickhouse_result": results,
                "error": None,
                "intermediate_steps": [HumanMessage(content=json.dumps(results, indent=2))],
            }
    except Exception as e:
        e = str(e).split("Stack trace")[0]
        return {
            "error": f"ClickHouse query error: {e}",
            "intermediate_steps": [HumanMessage(content=e)],
        }


def format_response(state: AgentState, llm) -> dict:
    prompt = """
You are a helpful assistant tasked with presenting data from a Clickhouse database in clear, concise, and natural language.

**Instructions:**
1. **Data Presentation:** Transform the provided data into a human-readable format. Focus on clarity and conciseness, highlighting key information.
2. **Empty Data Handling:** If the provided data is empty or null, respond with the phrase: "No results found."
3. **Markdown Formatting:** Utilize Markdown syntax to enhance readability and structure your response (e.g., headings, lists, tables).

**Input:**
- **Original User Query:** `{user_query}`
- **Retrieved Data:**
  {data}

**Output:**
Your response, formatted in Markdown.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder(variable_name="messages"), ("user", prompt)]
    )
    chain = prompt_template | llm
    result = chain.invoke(
        {
            "user_query": state["user_query"],
            "data": json.dumps(state["clickhouse_result"]),
            "sql_queries": state["sql_queries"],
            "messages": state.get("intermediate_steps", []),
        }
    )
    result = strip_thinking(result.content)
    return {"formatted_response": result, "intermediate_steps": [AIMessage(content=result)]}


def validate_response(state: AgentState, llm) -> dict:
    prompt = """
**Task:** Evaluate the correctness and helpfulness of a generated SQL query and its corresponding response given a user query and relevant context.

**Input:**

- **Original User Query:** `{user_query}`
- **Response:** `{response}`
- **SQL Queries:**
  {sql_queries}
- **Retrieved Data:**
  {data}

**Instructions:**

1. Analyze the provided `sql_queries` to determine if the SQL is syntactically correct and logically sound for answering the `User Query`. Consider potential edge cases and ambiguities.
2. Assess if the response generated by the SQL query and it's retrieved data is helpful and addresses the user's original question in a clear and informative way. Consider if the response is complete and avoids misleading information.
3. Reason step-by-step, explaining your analysis of the SQL query and its response in relation to the user query and schema. Clearly articulate your reasoning process.
4. Conclude your response with either "<answer>valid</answer>" or "<answer>invalid</answer>". "valid" indicates that the SQL query and its implied response are correct and helpful. "invalid" indicates that the SQL query is incorrect, the implied response is unhelpful, or both.

**Output:**

- Your evaluation of the SQL query and response.
- <answer>valid</answer> or <answer>invalid</answer>
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder(variable_name="messages"), ("user", prompt)]
    )
    chain = prompt_template | llm
    result = chain.invoke(
        {
            "user_query": state["user_query"],
            "response": state["formatted_response"],
            "messages": state.get("intermediate_steps", []),
            "sql_queries": "\n".join(state["sql_queries"]),
            "data": json.dumps(state["clickhouse_result"]),
        }
    )
    result = strip_thinking(result.content)
    is_invalid = "<answer>invalid</answer>" in result
    return {"is_valid": not is_invalid, "error": result if is_invalid else None}


def is_sql_fixed(state: AgentState) -> Literal["fix_sql", "execute_sql", "unable_to_fix_sql"]:
    if not state.get("error"):
        return "execute_sql"
    if state.get("error") == "Unable to fix SQL query.":
        return "unable_to_fix_sql"
    return "fix_sql"


def is_sql_successfully_executed(
    state: AgentState,
) -> Literal["fix_sql", "format_response", "unable_to_fix_sql"]:
    if not state.get("error") and state.get("clickhouse_result", []):
        return "format_response"
    if state.get("error") == "Unable to fix SQL query.":
        return "unable_to_fix_sql"
    return "fix_sql"


def is_response_valid(state: AgentState) -> Literal["fix_sql", "response_valid"]:
    if not state.get("error"):
        return "response_valid"
    return "fix_sql"


def build_graph(llm, debug: bool = True) -> CompiledGraph:
    graph = StateGraph(AgentState)
    graph.set_entry_point("generate_sql")

    # Add nodes
    graph.add_node("generate_sql", functools.partial(generate_sql_query, llm=llm))
    graph.add_node("fix_sql", functools.partial(fix_sql, llm=llm))
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("format_response", functools.partial(format_response, llm=llm))
    graph.add_node("validate_response", functools.partial(validate_response, llm=llm))

    # Add edges
    graph.add_edge("format_response", "validate_response")
    graph.add_conditional_edges(
        "generate_sql",
        is_sql_fixed,
        {"execute_sql": "execute_sql", "fix_sql": "fix_sql", "end": END},
    )
    graph.add_conditional_edges(
        "fix_sql", is_sql_fixed, {"execute_sql": "execute_sql", "unable_to_fix_sql": END}
    )
    graph.add_conditional_edges(
        "execute_sql",
        is_sql_successfully_executed,
        {"fix_sql": "fix_sql", "format_response": "format_response", "unable_to_fix_sql": END},
    )
    graph.add_conditional_edges(
        "validate_response", is_response_valid, {"fix_sql": "fix_sql", "response_valid": END}
    )

    # Compile and save graph
    graph = graph.compile(debug=debug)
    with open("graph.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
    return graph


@router.get("/dev/chat-with-db", summary="Global RateLimit: 2req/min if no Gemini API key provided")
def chat_with_db(
    req: Request, res: Response, prompt: str, gemini_api_key: str | None = None
) -> dict:
    limiter.apply_limits(
        req,
        res,
        "/v1/dev/chat-with-db",
        [RateLimit(limit=2, period=60)] if not gemini_api_key else [],
        [RateLimit(limit=2, period=60)] if not gemini_api_key else [],
        [RateLimit(limit=2, period=60)] if not gemini_api_key else [],
    )
    res.headers["Cache-Control"] = "public, max-age=3600"

    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("LLM_MODEL", "gemini-2.0-flash"),
        temperature=0.7,
        api_key=gemini_api_key or os.environ.get("GOOGLE_API_KEY"),
    )
    graph = build_graph(llm)
    inputs = {"user_query": prompt, "intermediate_steps": []}
    return graph.invoke(inputs, {"recursion_limit": 15})
