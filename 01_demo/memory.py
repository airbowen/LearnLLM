from langchain.memory import ConversationBufferMemory

# 创建一个 ConversationBufferMemory 实例，用于存储对话历史
memory = ConversationBufferMemory(memory_key="chat_history")

from langchain.prompts import PromptTemplate

# 定义对话模板，包括对话历史和问题
template_str = """
You are a chatbot having a conversation with a human.
Previous conversation:
{chat_history}
Human: {question}
AI:"""

# 根据模板创建 PromptTemplate 实例
prompt_template = PromptTemplate.from_template(template_str)

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# 创建一个基于 OpenAI 的聊天模型实例
llm = ChatOpenAI(temperature=0)

# 创建一个带有记忆功能的 LLMChain 实例
memory_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True, # verbose 表示打印详细过程
    memory=memory,
)

# 第一轮对话
memory_chain.predict(question="你好，我是jack")

# 第二轮对话
memory_chain.predict(question="你还记得我叫什么吗？")

"""
> Entering new LLMChain chain...
Prompt after formatting:

You are a chatbot having a conversation with a human.
Previous conversation:

Human: 你好，我是jack
AI:

> Finished chain.
 你好，jack。我是一个聊天机器人。你有什么需要我帮助的吗？
"""
memory_chain.predict(question="你还记得我叫什么吗？")
"""
> Entering new LLMChain chain...
Prompt after formatting:

You are a chatbot having a conversation with a human.
Previous conversation:
Human: 你好，我是jack
AI:  你好，jack。我是一个聊天机器人。你有什么需要我帮助的吗？
Human: 你还记得我叫什么吗？
AI:

> Finished chain.
 当然，你刚刚告诉我你叫jack。你有什么其他问题吗？
"""
#在后续的对话中，发送给 LLM 的提示词都带上了历史对话内容，LLM 也确实做出令人满意的回答。