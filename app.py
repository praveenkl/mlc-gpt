import os, sys
import json
import logging
import gradio as gr

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from llama_index.core import Settings, SQLDatabase, VectorStoreIndex
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import VectorStoreInfo
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, RouterQueryEngine
from llama_index.core.query_engine import SQLAutoVectorQueryEngine
from llama_index.llms.openai import OpenAI
from sqlalchemy import create_engine, MetaData
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.selectors import LLMSingleSelector
import chromadb
import openai

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
openai.api_key = os.environ["OPENAI_API_KEY"]
Settings.llm = OpenAI(model="gpt-4-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

def get_table_names(engine):
    """Get all table names from a database."""
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table_names = []
    for table in metadata.tables.values():
        table_names.append(table.name)
    return table_names

def get_sql_tool(year):
    """Get a tool for translating natural language queries into SQL queries."""
    engine = create_engine(f'sqlite:///data/stats/mlc_stats_{year}.db')
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
    obj_index.as_retriever(similarity_top_k=1),
    )
    st = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        description=(
            "Useful for translating a natural language query into a SQL query over"
            " a database named mlc_stats to answer statistics related questions about major league cricket or MLC. The database contains 4 tables:"
            " batting_statistics, bowling_statistics, team_statistics, and match_details. batting_statistics table contains"
            " batting related information about each player, bowling_statistics table contains"
            " bowling related information about each player, team_statistics table contains information about each"
            " team, and match_details table contains information about each match." 
        ),
    )
    return st

def get_index(year):
    """Get a vector store index for the chroma database."""
    db = chromadb.PersistentClient(path="./data/news/chroma_db")
    try:
        coll_name = f"mlc_articles_{year}"
        chroma_collection = db.get_collection(coll_name)
    except ValueError:
        return None
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection, persist_dir=f'./data/news/storage/{year}')
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index

def get_vector_tool(year):
    """Get a tool for translating natural language queries into vector queries."""
    index = get_index(year)
    if index is None:
        print("Chroma collection not found. Please index documents and try again.")
        sys.exit(1)

    vector_store_info = VectorStoreInfo(
      content_info="articles about major league cricket or MLC",
        metadata_info=[]
    )
    
    vector_auto_retriever = VectorIndexAutoRetriever(
        index, vector_store_info=vector_store_info,
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
        similarity_top_k=4,
    )
    retriever_query_engine = RetrieverQueryEngine.from_args(vector_auto_retriever)

    vt = QueryEngineTool.from_defaults(
       query_engine=retriever_query_engine,
        description=(
        "Useful for answering semantic questions about major league cricket or MLC"
        ),
    )
    return vt

def get_query_engine_tool(year):
    """Get a query engine tool for answering questions about major league cricket."""
    sql_tool = get_sql_tool(year)
    vector_tool = get_vector_tool(year)
    qe = SQLAutoVectorQueryEngine(sql_tool, vector_tool)

    qe_description = f'Current year is 2024. Useful for answering questions about the {year} edition of the Major League Cricket (MLC) tournament.'
    if year == 2024:
        qe_description += ' Also useful for answering questions about Major League Cricket when the year cannot be determined from the question.'
    qe_tool = QueryEngineTool.from_defaults(
        query_engine=qe,
        description=qe_description,
    )
    return qe_tool

def get_completed_matches():
    """Get the completed matches."""
    with open("data/match_reports/2024/completed_matches.json", "r") as f:
        schedule = json.load(f)
    return schedule

def get_schedule():
    """Get the schedule of matches."""
    with open("data/match_reports/2024/schedule.json", "r") as f:
        schedule = json.load(f)
    return schedule

def get_match_report(match_id):
    """Get the match report."""
    with open(f"data/match_reports/2024/{match_id}_report.txt", "r") as f:
        report = f.read()
    return report

if __name__ == '__main__':
    # my_theme = gr.themes.Base()
    my_theme = gr.themes.Soft(spacing_size=gr.themes.sizes.spacing_sm, text_size=gr.themes.sizes.text_sm)

    qe_tool_2023 = get_query_engine_tool(2023)
    qe_tool_2024 = get_query_engine_tool(2024)    
    query_engine = RouterQueryEngine(selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            qe_tool_2023,
            qe_tool_2024,
        ],
    )

    def handle_query(query):
        """Handle the query."""
        if len(query) > 200:
            response = "Sorry, your query is too long. Please try again with a shorter query."
            return response
        try:
            print(f"Question: {query}")
            response = query_engine.query(query)
        except Exception as e: # pylint: disable=broad-exception-caught
            print(e)
            response = "Sorry, an error ocurred while answering the query. Please try again later."
        print(f"Answer: {response}")
        return response

    def handle_click(l, b):
        if b == "Show Report":
            print(f"Show report for {l}")
            t = gr.Textbox(visible=True)
            b = gr.Button(value="Hide Report")
        else:
            t = gr.Textbox(visible=False)
            b = gr.Button(value="Show Report")
        return b, t

    demo1 = gr.Interface(
        fn=handle_query,
        inputs=gr.Textbox(lines=2, placeholder="Enter query here..."),
        outputs=gr.Textbox(),
        examples=[
        ["Who is the owner of Major league cricket and how much does it cost to run the league?"],
        ["What's different about this years edition of MLC?"],
        ["How many total sixes were scored by the batsmen in the tournament and which team scored the most?"],
        ["How many batsmen scored centuries and what are their names?"],
        ["What are the names of teams that won matches played in Church Street Park ground?"]
        ],
        cache_examples=True,
        theme=my_theme,
    )

    with gr.Blocks(theme=my_theme) as demo2:
        match_info = get_completed_matches()
        for id, details in match_info.items():
            [match_num, teams, date, city, result] = details
            report = ""
            if len(result) > 0:
                report = get_match_report(id)
            with gr.Row(variant="compact"):
                l = gr.Label(value=teams, label=f'Match {match_num} | {date} | {city}', container=True, scale=16)
                b = gr.Button(value="Show Report", scale=1, size="sm")
            t = gr.Textbox(lines=7, value=report, interactive=False, show_copy_button=True, visible=False, label="Match Report")
            b.click(fn=handle_click, inputs=[l, b], outputs=[b,t])

    with gr.Blocks(theme=my_theme) as demo3:
        match_info = get_schedule()
        for id, details in match_info.items():
            [match_num, teams, date, city, result] = details
            with gr.Row(variant="compact"):
                l = gr.Label(value=teams, label=f'Match {match_num} | {date} | {city}', container=True)

    demo = gr.TabbedInterface([demo1, demo2, demo3],
                              tab_names=["Ask Questions", "Read Reports", "View Schedule"],
                              title="MLC Guru",
                              analytics_enabled=False,
                              theme=my_theme,)

    if "PORT" in os.environ:
        port = int(os.environ["PORT"])
        demo.launch(share=False, server_name="0.0.0.0", server_port=port)
    else:
        demo.launch(share=False)
    
    # while True:
    #     q = input("Enter query (or quit): ")
    #     if q.lower() == "quit":
    #         break
    #     response = handle_query(q)
    #     print(response)
        # print(response.metadata)
