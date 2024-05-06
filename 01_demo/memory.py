from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")
from langchain.prompts import PromptTemplate

template_str = """
You are a chatbot having a conversation with a human.
Previous conversation:
{chat_history}
Human: {question}
AI:"""

prompt_template = PromptTemplate.from_template(template_str)
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

llm = ChatOpenAI(temperature=0)
memory_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True, # verbose 表示打印详细过程
    memory=memory,
)
memory_chain.predict(question="你好，我是jack")
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
