import os, sys
import json
import logging
import re
import gradio as gr

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from llama_index.core import Settings, SQLDatabase, VectorStoreIndex, PromptTemplate
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import VectorStoreInfo
from llama_index.core.retrievers import VectorIndexAutoRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, RouterQueryEngine
from llama_index.core.query_engine import SQLAutoVectorQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.llms.openai import OpenAI
from sqlalchemy import create_engine, MetaData
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.postprocessor import LLMRerank
import chromadb
import openai

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
openai.api_key = os.environ["OPENAI_API_KEY"]
Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

def get_table_names(engine):
    """Get all table names from a database."""
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table_names = []
    for table in metadata.tables.values():
        table_names.append(table.name)
    return table_names

class CustomSQLParser():
    """Custom SQL Parser."""

    def parse_response_to_sql(self, response: str, query_bundle: QueryBundle) -> str:
        """Parse response to SQL."""
        sql_query_start = response.find("SQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            if response.startswith("SQLQuery:"):
                response = response[len("SQLQuery:") :]
        sql_result_start = response.find("SQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start]
        response = response.replace("```sql", "").replace("```", "").strip()
        newline_split = re.split(r'\n+', response, maxsplit=1)
        response = newline_split[0].strip()
        return response

def get_sql_query_engine(year, league="major"):
    engine = create_engine(f'sqlite:///data/{league}/stats/stats_{year}.db')
    sql_database = SQLDatabase(engine=engine)
    table_node_mapping = SQLTableNodeMapping(sql_database)
    all_table_names = get_table_names(engine)
    table_schema_objs = []
    for table_name in all_table_names:
        table_schema_objs.append(SQLTableSchema(table_name=table_name))
    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )
    sql_query_engine = SQLTableRetrieverQueryEngine(
    sql_database,
    obj_index.as_retriever(similarity_top_k=10),
    )
    # Disable handling of SQL errors to prevent asking the LLM to fix them
    sql_query_engine._sql_retriever._handle_sql_errors = False
    sql_query_engine._sql_retriever._verbose = True

    # Add custom SQL parser to workaround a bug in LlamaIndex's default parser
    sql_query_engine._sql_retriever._sql_parser = CustomSQLParser()

    # Add year context to the default text-to-SQL prompt
    prompts = sql_query_engine.get_prompts()
    current_text_to_sql_prompt = prompts["sql_retriever:text_to_sql_prompt"]
    current_text_to_sql_prompt_str = current_text_to_sql_prompt.get_template()
    prompt_prefix = f'Current year is 2024.\n\n'
    new_text_to_sql_prompt_str = prompt_prefix + current_text_to_sql_prompt_str
    new_text_to_sql_prompt = PromptTemplate(new_text_to_sql_prompt_str)
    sql_query_engine.update_prompts({"sql_retriever:text_to_sql_prompt": new_text_to_sql_prompt})

    return sql_query_engine

def get_sql_tool(year, include_year_context=False, league="major"):
    """Get a tool for translating natural language queries into SQL queries."""
    sql_query_engine = get_sql_query_engine(year, league)
    st = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        description=(
            "Useful for translating a natural language query into a SQL query over"
            " a database named stats to answer statistics related questions about a cricket league. The database contains 4 tables:"
            " batting_statistics, bowling_statistics, team_statistics, and match_details. batting_statistics table contains"
            " batting related information about each player, bowling_statistics table contains"
            " bowling related information about each player, team_statistics table contains information about each"
            " team, and match_details table contains information about each match." 
        ),
    )
    if include_year_context:
        qe_description = get_year_context(year, league)
        updated_description = qe_description + "\n\n" + st.metadata.description
        st.metadata.description = updated_description
    return st

def get_index(year, league="major"):
    """Get a vector store index for the chroma database."""
    db = chromadb.PersistentClient(path=f"./data/{league}/news/chroma_db")
    if league == "major":
        coll_name = f"mlc_articles_{year}"
    else:
        if year == 2025:
            coll_name = f"milc_articles_2024"
        else:
           coll_name = f"milc_articles_{year}"
    try:
        chroma_collection = db.get_collection(coll_name)
    except ValueError:
        return None
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection, persist_dir=f'./data/{league}/news/storage/{year}')
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index

def get_vector_query_engine(year, auto_retriever, league="major"):
    index = get_index(year, league)
    if index is None:
        print("Chroma collection not found. Please index documents and try again.")
        sys.exit(1)

    vector_store_info = VectorStoreInfo(
      content_info=f"articles about {league} league cricket",
        metadata_info=[]
    )
    
    if auto_retriever:
        vector_retriever = VectorIndexAutoRetriever(
            index, vector_store_info=vector_store_info,
            vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
            similarity_top_k=20)
    else:
        vector_retriever = VectorIndexRetriever(
            index, vector_store_info=vector_store_info,
            vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
            similarity_top_k=20)
    
    pp = LLMRerank(top_n=4, choice_batch_size=20)
    retriever_query_engine = RetrieverQueryEngine.from_args(vector_retriever, node_postprocessors=[pp])
    
    return retriever_query_engine

def get_vector_tool(year, include_year_context=False, auto_retriever=True, league="major"):
    """Get a tool for translating natural language queries into vector queries."""
    retriever_query_engine = get_vector_query_engine(year, auto_retriever, league)
    vt = QueryEngineTool.from_defaults(
       query_engine=retriever_query_engine,
        description=(
        f"Useful for answering semantic questions about {league} league cricket"
        ),
    )
    if include_year_context:
        qe_description = get_year_context(year, league)
        updated_description = qe_description + "\n\n" + vt.metadata.description
        vt.metadata.description = updated_description
    return vt

def get_dynamic_query_engine_tool(year, league="major"):
    """Get a query engine tool for answering questions about a cricket league."""
    sql_tool = get_sql_tool(year, include_year_context=False, league=league)
    vector_tool = get_vector_tool(year, auto_retriever=True, include_year_context=False, league=league)
    qe = SQLAutoVectorQueryEngine(sql_tool, vector_tool)
    qe_description = get_year_context(year, league)
    qe_tool = QueryEngineTool.from_defaults(
        query_engine=qe,
        description=qe_description,
    )
    return qe_tool

def get_query_engine(type, league):
    if type == "dynamic":
        tool_2023 = get_dynamic_query_engine_tool(2023, league)
        tool_2024 = get_dynamic_query_engine_tool(2024, league)
        tool_2025 = get_dynamic_query_engine_tool(2025, league)
        query_engine = RouterQueryEngine(selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[
                tool_2023,
                tool_2024,
                tool_2025,
            ],
        )
    elif type == "stats":
        tool_2023 = get_sql_tool(2023, include_year_context=True, league=league)
        tool_2024 = get_sql_tool(2024, include_year_context=True, league=league)
        tool_2025 = get_sql_tool(2025, include_year_context=True, league=league)
        query_engine = RouterQueryEngine(selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[
                tool_2023,
                tool_2024,
                tool_2025,
            ],
        )
    else:
        tool_2023 = get_vector_tool(2023, auto_retriever=False, include_year_context=True, league=league)
        tool_2024 = get_vector_tool(2024, auto_retriever=False, include_year_context=True, league=league)
        tool_2025 = get_vector_tool(2025, auto_retriever=False, include_year_context=True, league=league)
        query_engine = RouterQueryEngine(selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[
                tool_2023,
                tool_2024,
                tool_2025,
            ],
        )
    return query_engine

def get_year_context(year, league="major"):
    """Get the year context for the query engine tool."""
    qe_description = f'Current year is 2025. Useful for answering questions about the {year} edition of the {league} League Cricket tournament.'
    if year == 2025:
        qe_description += f' Also useful for answering questions about {league} League Cricket when the year cannot be determined from the question.'
    return qe_description

def get_completed_matches(league="major"):
    """Get the completed matches."""
    with open(f"data/{league}/match_reports/2024/completed_matches.json", "r") as f:
        schedule = json.load(f)
    return schedule

def get_schedule(league="major"):
    """Get the schedule of matches."""
    with open(f"data/{league}/match_reports/2024/schedule.json", "r") as f:
        schedule = json.load(f)
    return schedule

def get_match_report(match_id, league="major"):
    """Get the match report."""
    if league == "major":
        with open(f"data/{league}/match_reports/2024/{match_id}_report.txt", "r") as f:
            report = f.read()
    else:
        with open(f"data/{league}/match_reports/2024/{match_id}.txt", "r") as f:
            report = f.read()
    return report

def get_city(ground):
    """Get the city based on the ground."""
    if ground == "Church Street Park":
        city = "Morrisville"
    else:
        city = "Dallas"
    return city


if __name__ == '__main__':
    my_theme = gr.themes.Soft(spacing_size=gr.themes.sizes.spacing_sm, text_size=gr.themes.sizes.text_sm)

    # Define a map of query engines based on type and league
    query_engines = {
        "dynamic": {
            "major": get_query_engine("dynamic", "major"),
            "minor": get_query_engine("dynamic", "minor"),
        },
        "stats": {
            "major": get_query_engine("stats", "major"),
            "minor": get_query_engine("stats", "minor"),
        },
        "news": {
            "major": get_query_engine("news", "major"),
            "minor": get_query_engine("news", "minor"),
        },
    }
    
    def handle_query(query, league="major", type="dynamic"):
        """Handle the query."""

        # Check if the query is too long
        if len(query) > 200:
            response = "Sorry, your query is too long. Please try again with a shorter query."
            return response
        try:
            print(f"Question: {query}")
            print(f"League: {league}")
            response = query_engines[type][league].query(query)
        # catch list index out of range error
        except IndexError as e:
            print(e)
            response = "The provided context does not contain enough information to answer the question."
        except KeyError as e:
            print(e)
            response = "Sorry, an error occured while selecting the appropriate query engine. Please try again later."
        except Exception as e: # pylint: disable=broad-exception-caught
            print(e)
            response = "Sorry, an error ocurred while answering the query. Please try again later."
        print(f"Answer: {response}")
        return response, league

    def handle_click(l, b):
        if b == "Show Report":
            print(f"Show report for {l}")
            t = gr.Textbox(visible=True)
            b = gr.Button(value="Hide Report")
        else:
            t = gr.Textbox(visible=False)
            b = gr.Button(value="Show Report")
        return b, t

    # Define "Ask Questions" block
    def create_ask_questions_block(league):
        return gr.Interface(
            fn=handle_query,
            inputs=[
                gr.Textbox(lines=2, placeholder="Enter query here...", show_label=False),
                gr.State(value=league),
                # gr.Radio(choices=["stats", "news", "dynamic"], value="dynamic", show_label=False, container=True),
            ],
            outputs=[gr.Textbox(show_label=False), gr.State()],
            examples=[
                ["What are the teams? Who owns them?"],
                ["What's different about this years edition of the league?"],
                ["How many total sixes were scored by the batsmen in the tournament and which team scored the most?"],
                ["How many batsmen scored centuries and what are their names?"],
            ],
            cache_examples=False,
            flagging_mode="never",
            theme=my_theme,
        )


    # Define "Read Reports" block
    def create_read_reports_block(league):
        with gr.Blocks(theme=my_theme) as read_reports:
            match_info = get_completed_matches(league)
            for id, details in match_info.items():
                [match_num, teams, date, ground, result] = details
                city = get_city(ground)
                report = ""
                if len(result) > 0:
                    report = get_match_report(id, league)
                with gr.Row(variant="compact", equal_height=True):
                    l = gr.Label(value=teams, label=f"Match {match_num} | {date} | {city}", container=True, scale=16)
                    b = gr.Button(value="Show Report", scale=1, size="sm")
                t = gr.Textbox(
                    lines=7,
                    value=report,
                    interactive=False,
                    show_copy_button=True,
                    visible=False,
                    label="Match Report",
                )
                b.click(fn=handle_click, inputs=[l, b], outputs=[b, t])
            gr.HTML("<hr style='border: 2px solid #808080;'>")
            match_info = get_schedule(league)
            for id, details in match_info.items():
                [match_num, teams, date, ground, result] = details
                city = get_city(ground)
                with gr.Row(variant="compact", equal_height=True):
                    gr.Label(value=teams, label=f"Match {match_num} | {date} | {city}", container=True)
        return read_reports


    def create_tabbed_interface(league):
        # Create Ask Questions Block
        ask_questions_block = create_ask_questions_block(league)

        # Create Read Reports Block
        read_reports_block = create_read_reports_block(league)

        # Combine into Tabbed Interface
        tabbed_interface = gr.TabbedInterface(
            [ask_questions_block, read_reports_block],
            tab_names=["Ask Questions", "Read Reports"],
            theme=my_theme,
        )
        return tabbed_interface

    # Create Major League Tabs
    league_one_tabs = create_tabbed_interface("major")

    # Create Minor League Tabs
    league_two_tabs = create_tabbed_interface("minor")

    # Combine into Top-Level Tabbed Interface
    top_level_interface = gr.TabbedInterface(
        [league_one_tabs, league_two_tabs],
        tab_names=["Major League", "Minor League"],
        theme=my_theme,
        title="MLC Guru"
    )

    # Launch the App
    if "PORT" in os.environ:
        port = int(os.environ["PORT"])
        top_level_interface.launch(share=False, server_name="0.0.0.0", server_port=port)
    else:
        top_level_interface.launch(share=False)
