import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
google_api_key = os.environ["GOOGLE_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser

class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="The name of the person")
    lastname: Optional[str] = Field(
        default=None, description="The lastname of the person if known"
    )
    country: Optional[str] = Field(
        default=None, description="The country of the person if known"
    )

# Create a parser for Person
person_parser = JsonOutputParser(pydantic_object=Person)

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt_person = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value. "
            "Return the output as a JSON object conforming to the following schema:\n{format_instructions}",
        ),
        ("human", "{text}"),
    ]
).partial(format_instructions=person_parser.get_format_instructions())

chain_person = prompt_person | llm | person_parser

comment_person = "I absolutely love this product! It's been a game-changer for my daily routine. The quality is top-notch and the customer service is outstanding. I've recommended it to all my friends and family. - Sarah Johnson, USA"

response_person = chain_person.invoke({"text": comment_person})

print("\n----------\n")

print("Key data extraction (single person):")

print("\n----------\n")
print(response_person)

print("\n----------\n")

class Data(BaseModel):
    """Extracted data about people."""
    people: List[Person] = Field(description="A list of people extracted from the text.")
    
# Create a parser for Data
data_parser = JsonOutputParser(pydantic_object=Data)

# Define a custom prompt to provide instructions and any additional context.
# This prompt will be used for the Data schema.
prompt_data = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value. "
            "Return the output as a JSON object conforming to the following schema:\n{format_instructions}",
        ),
        ("human", "{text}"),
    ]
).partial(format_instructions=data_parser.get_format_instructions())

chain_data = prompt_data | llm | data_parser

comment_data = "I'm so impressed with this product! It has truly transformed how I approach my daily tasks. The quality exceeds my expectations, and the customer support is truly exceptional. I've already suggested it to all my colleagues and relatives. - Emily Clarke, Canada"

response_data = chain_data.invoke({"text": comment_data})

print("\n----------\n")

print("Key data extraction of a list of entities (single comment):")

print("\n----------\n")
print(response_data)

print("\n----------\n")

# Example input text that mentions multiple people
text_input_multiple = """
Alice Johnson from Canada recently reviewed a book she loved. Meanwhile, Bob Smith from the USA shared his insights on the same book in a different review. Both reviews were very insightful.
"""

# Invoke the processing chain on the text
response_multiple = chain_data.invoke({"text": text_input_multiple})

# Output the extracted data
print("\n----------\n")

print("Key data extraction of a review with several users:")

print("\n----------\n")
print(response_multiple)

print("\n----------\n")
