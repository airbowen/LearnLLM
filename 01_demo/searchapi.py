import os
from langchain_community.utilities import SearchApiAPIWrapper
from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.runnables import RunnablePassthrough, Runnable
from langchain.templates import ChatPromptTemplate
from langchain.interfaces import BaseLanguageModel
from langchain_openai import OpenAIToolsAgentOutputParser
from typing import Sequence

# Set the environment variable, replace with your own API key, register at: https://searchapi.com
# 设置环境变量，替换成自己的 API 密钥，注册地址：https://searchapi.com
os.environ["SEARCHAPI_API_KEY"] = "替换成自己的 api key"

# Initialize SearchApiAPIWrapper
# 初始化 SearchApiAPIWrapper
search = SearchApiAPIWrapper()

# Get information about LangChain using SearchApiAPIWrapper
# 使用 SearchApiAPIWrapper 获取关于 LangChain 的信息
print(search.run("什么是langchain"))

# Create a search tool Tool
# 创建一个搜索工具 Tool
search_tool = Tool(
    name="search_tool",
    func=search.run,
    description="Useful for when you need to ask with search",
)

# Local file management tool
# 本地文件管理工具
# Set the file management root directory to "/data/"
# 设置文件管理根目录为 "/data/"
tools = FileManagementToolkit(root_dir="/data/").get_tools()

# Select the WriteFileTool for writing files
# 选择写入文件的工具 WriteFileTool
write_file_tool = tools[5]

# Write to a file using WriteFileTool
# 使用 WriteFileTool 写入文件
write_file_tool.invoke({"file_path": "example.txt", "text": "LangChain"})

# Initialize OpenAI's GPT-3.5 model
# 初始化 OpenAI 的 GPT-3.5 模型
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Get the dialogue template
# 获取对话模板
prompt = hub.pull("hwchase17/openai-tools-agent")

def create_openai_tools_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate
) -> Runnable:
    # Bind tools to the language model
    # 将工具绑定到语言模型上
    llm_with_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

    # Create the OpenAI tools agent
    # 创建 OpenAI 工具代理
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            )
        )
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    return agent

# Create the OpenAI tools agent
# 创建 OpenAI 工具代理
openai_tools_agent = create_openai_tools_agent(llm, tools, prompt)
