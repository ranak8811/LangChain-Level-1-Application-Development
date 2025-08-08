import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
google_api_key = os.environ["GOOGLE_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    political_tendency: str = Field(
        description="The political tendency of the user"
    )
    language: str = Field(description="The language the text is written in")

# Create a parser for the first Classification schema
classification_parser_1 = JsonOutputParser(pydantic_object=Classification)

# Example for few-shot prompting
example_input = "This movie was fantastic! I loved every minute of it. It's a masterpiece. - John Doe, USA"
example_output_classification = {
    "sentiment": "positive",
    "political_tendency": "neutral",
    "language": "english"
}

tagging_prompt_1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value. "
            "Return the output as a JSON object conforming to the following schema:\n{format_instructions}\n\n"
            "Here is an example:\n"
            "Passage: {example_input}\n"
            "Output: {example_output_classification}\n"
        ),
        ("human", "{input}"),
    ]
).partial(
    format_instructions=classification_parser_1.get_format_instructions(),
    example_input=example_input,
    example_output_classification=example_output_classification
)

tagging_chain_1 = tagging_prompt_1 | llm | classification_parser_1

trump_follower = "I'm confident that President Trump's leadership and track record will once again resonate with Americans. His strong stance on economic growth and national security is exactly what our country needs at this pivotal moment. We need to bring back the proven leadership that can make America great again!"

biden_follower = "I believe President Biden's compassionate and steady approach is vital for our nation right now. His commitment to healthcare reform, climate change, and restoring our international alliances is crucial. It's time to continue the progress and ensure a future that benefits all Americans."

response_trump_1 = tagging_chain_1.invoke({"input": trump_follower})

print("\n----------\n")

print("Sentiment analysis Trump follower (basic):")

print("\n----------\n")
print(response_trump_1)

print("\n----------\n")

response_biden_1 = tagging_chain_1.invoke({"input": biden_follower})

print("\n----------\n")

print("Sentiment analysis Biden follower (basic):")

print("\n----------\n")
print(response_biden_1)

print("\n----------\n")

class ClassificationWithEnums(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    political_tendency: str = Field(
        ...,
        description="The political tendency of the user",
        enum=["conservative", "liberal", "independent"],
    )
    language: str = Field(
        ..., enum=["spanish", "english"]
    )
    
# Create a parser for the second Classification schema (with enums)
classification_parser_2 = JsonOutputParser(pydantic_object=ClassificationWithEnums)

example_output_classification_enums = {
    "sentiment": "happy",
    "political_tendency": "independent",
    "language": "english"
}

tagging_prompt_2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value. "
            "Return the output as a JSON object conforming to the following schema:\n{format_instructions}\n\n"
            "Here is an example:\n"
            "Passage: {example_input}\n"
            "Output: {example_output_classification_enums}\n"
        ),
        ("human", "{input}"),
    ]
).partial(
    format_instructions=classification_parser_2.get_format_instructions(),
    example_input=example_input,
    example_output_classification_enums=example_output_classification_enums
)

tagging_chain_2 = tagging_prompt_2 | llm | classification_parser_2

response_trump_2 = tagging_chain_2.invoke({"input": trump_follower})

print("\n----------\n")

print("Sentiment analysis Trump follower (with a list of options using enums):")

print("\n----------\n")
print(response_trump_2)

print("\n----------\n")

response_biden_2 = tagging_chain_2.invoke({"input": biden_follower})

print("\n----------\n")

print("Sentiment analysis Biden follower (with a list of options using enums):")

print("\n----------\n")
print(response_biden_2)

print("\n----------\n")
