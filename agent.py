from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# *** 关键更改：切换回 HuggingFaceEndpoint，解决 InferenceClient 错误 ***
from langchain_huggingface import HuggingFaceEndpoint
from tools import get_tools


# 接收 Hugging Face API Key
def build_garbage_agent(hf_api_key: str):
    # 1. 加载工具
    tools = get_tools()

    # 2. 初始化大模型 (使用 Hugging Face Endpoint)
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-2-7b-chat-hf", # 使用 Llama-2 聊天模型，指令遵循能力强
        temperature=0.0, # 保持低温度以提高稳定性
        max_new_tokens=1024, # 使用 max_new_tokens 参数 (HuggingFaceEndpoint 专用)
        # *** 关键修复：添加停止序列以防止模型输出干扰 Agent 解析器 ***
        # ReAct 模式在生成 Action 后，期望接收到 Observation。
        # 强制模型在看到 Observation 的起始字符时停止。
        stop_sequences=["\nObservation:"],
        # 将密钥作为 Token 传入
        huggingfacehub_api_token=hf_api_key
    )

    # 3. 获取标准的 ReAct Prompt 模板
    # 这个模板包含了: Thought, Action, Action Input, Observation
    prompt = hub.pull("hwchase17/react")

    # 自定义 System Prompt，让它专注于垃圾分类
    base_prompt = """
    你是一个专业的“垃圾分类智能助手”。
    你的目标是帮助用户准确地将物品进行垃圾分类。

    请遵循以下步骤 (ReAct 模式):
    1. 观察用户输入（可能是图片路径或文字）。
    2. 如果是图片，必须先使用 'image_recognition_tool' 识别物品是什么。
    3. 知道物品是什么后，思考它属于哪类垃圾。如果不知道（或者规则因城市而异），必须使用 'web_search_tool' 查询（例如搜索 '上海 电池 垃圾分类'）。
    4. 综合信息，给出最终的分类建议和投放指导。
    """

    # 将我们的指令注入到 Prompt 中
    prompt.template = base_prompt + "\n" + prompt.template

    # 4. 创建 Agent
    agent = create_react_agent(llm, tools, prompt)

    # 5. 创建执行器 (verbose=True 可以看到思考过程，类似视频中的演示效果)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    return agent_executor