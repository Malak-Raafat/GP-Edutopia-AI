from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

print(wikipedia_tool.name) 
print(wikipedia_tool.description)

print(wikipedia_tool.run("Trump"))