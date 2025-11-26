import os
from langchain.tools import tool
# 确保安装了 langchain-community，否则 DuckDuckGoSearchRun 无法导入
from langchain_community.tools import DuckDuckGoSearchRun

# 初始化搜索工具
search = DuckDuckGoSearchRun()

@tool
def image_recognition_tool(image_path: str) -> str:
    """
    模拟图像识别工具。输入图片路径，返回图片内容的文本描述。
    """
    # 模拟识别结果，你需要确保这个函数被定义
    if "bottle" in image_path:
        return "图片中是一只空的塑料矿泉水瓶，瓶盖已经拧紧。"
    elif "apple" in image_path:
        return "图片中是一个吃剩的苹果核。"
    elif "battery" in image_path:
        return "图片中是一节废旧的5号干电池。"
    else:
        return "无法识别图片内容，请提供更清晰的图片。"

@tool
def web_search_tool(query: str) -> str:
    """
    网络搜索工具。当不确定某个物品属于哪类垃圾，或者需要查询特定城市规则时使用。
    """
    print(f"\n[Tool Call] 正在搜索: {query} ...")
    return search.run(query)

# 🚀 关键：必须有这个函数来导出工具列表
def get_tools():
    """
    导出所有可用的 Agent 工具。
    """
    return [image_recognition_tool, web_search_tool]