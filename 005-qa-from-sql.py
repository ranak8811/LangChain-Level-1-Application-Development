import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
google_api_key = os.environ["GOOGLE_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

from langchain_community.utilities import SQLDatabase

sqlite_db_path = "data/street_tree_db.sqlite"

db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")

from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnableLambda

# Define a runnable to strip the "SQLQuery: " prefix
strip_prefix = RunnableLambda(lambda x: x.replace("SQLQuery: ", "").strip())

chain = create_sql_query_chain(llm, db)

response_query_only = chain.invoke({"question": "List the species of trees that are present in San Francisco"})

print("\n----------\n")

print("Generated SQL Query:")

print("\n----------\n")
print(response_query_only)

print("\n----------\n")

print("Query executed:")

print("\n----------\n")

# Execute the stripped query
print(db.run(strip_prefix.invoke(response_query_only)))

print("\n----------\n")

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

execute_query = QuerySQLDataBaseTool(db=db)

write_query = create_sql_query_chain(llm, db)

# Chain for writing and executing query
chain_write_execute = write_query | strip_prefix | execute_query

response_write_execute = chain_write_execute.invoke({"question": "List the species of trees that are present in San Francisco"})

print("\n----------\n")

print("List the species of trees that are present in San Francisco (with query execution included):")

print("\n----------\n")
print(response_write_execute)

print("\n----------\n")

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

# Chain for full QA with SQL
chain_full_qa = (
    RunnablePassthrough.assign(query=write_query | strip_prefix).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

response_full_qa = chain_full_qa.invoke({"question": "List the species of trees that are present in San Francisco"})

print("\n----------\n")

print("List the species of trees that are present in San Francisco (passing question and result to the LLM):")

print("\n----------\n")
print(response_full_qa)

print("\n----------\n")
