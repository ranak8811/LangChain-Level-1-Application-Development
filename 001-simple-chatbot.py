import warnings

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
google_api_key = os.environ["GOOGLE_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI

chatbot = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

from langchain_core.messages import HumanMessage

messagesToTheChatbot = [
    HumanMessage(content="My favorite color is blue."),
]

response = chatbot.invoke(messagesToTheChatbot)

print("\n----------\n")

print("My favorite color is blue.")

print("\n----------\n")
print(response.content)

print("\n----------\n")

response = chatbot.invoke([
    HumanMessage(content="What is my favorite color?")
])

print("\n----------\n")

print("What is my favorite color?")

print("\n----------\n")
print(response.content)

print("\n----------\n")

from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = (
    RunnablePassthrough.assign(messages=lambda x: memory.load_memory_variables({})["messages"])
    | prompt
    | chatbot
    | StrOutputParser()
)

response = chain.invoke({"content": "hello!"})
memory.save_context({"input": "hello!"}, {"output": response})

print("\n----------\n")

print("hello!")

print("\n----------\n")
print(response)

print("\n----------\n")

response = chain.invoke({"content": "my name is Julio"})
memory.save_context({"input": "my name is Julio"}, {"output": response})

print("\n----------\n")

print("my name is Julio")

print("\n----------\n")
print(response)

print("\n----------\n")

response = chain.invoke({"content": "what is my name?"})
memory.save_context({"input": "what is my name?"}, {"output": response})

print("\n----------\n")

print("what is my name?")

print("\n----------\n")
print(response)

print("\n----------\n")
