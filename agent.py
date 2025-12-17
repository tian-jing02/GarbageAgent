# ------------------------------------------------------------------
# Agent 核心逻辑：使用本地 Ollama 模型 (RAG 优先策略)
# ------------------------------------------------------------------
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.chat_models import ChatOllama
from tools import get_tools


def build_garbage_agent():
    # 1. 加载工具
    tools = get_tools()

    # 2. 初始化本地大模型
    # 请根据你实际运行的模型名称设置
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.1,
    )

    # 3. 获取标准的 ReAct Prompt 模板
    prompt = hub.pull("hwchase17/react")

    # 自定义 System Prompt，强调 RAG 优先
    base_prompt = """
    你是一个专业的“垃圾分类智能助手”。
    你的目标是帮助用户准确地将物品进行垃圾分类。

    请遵循以下 ReAct 步骤 (从上到下优先级递减):

    1. **图像识别 (Image Recognition Tool):** 如果用户输入包含图片，必须先使用 'image_recognition_tool' 识别物品名称。

    2. **知识库检索 (Knowledge Retrieval Tool):** 识别出物品名称后，或者用户直接输入物品名称后，**必须** 优先使用 'knowledge_retrieval_tool' 在本地知识库中查找分类结果。

    3. **网络搜索 (Web Search Tool):** 只有当 'knowledge_retrieval_tool' 返回“未找到”或用户询问特定的城市分类规则（例如“上海的...”）时，才使用 'web_search_tool' 联网查询。

    4. 综合所有信息，给出最终的分类建议和投放指导。
    """

    # 将我们的指令注入到 Prompt 中
    prompt.template = base_prompt + "\n" + prompt.template

    # 4. 创建 Agent
    agent = create_react_agent(llm, tools, prompt)

    # 5. 创建执行器
    # max_iterations=4: 限制 Agent 最多思考/行动4次，防止陷入死循环，提高响应速度
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=4
    )

    return agent_executor