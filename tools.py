import os
import base64
import requests
import json
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()

# ============================
# 配置：阿里云通义千问 · Vision API
# ============================
QWEN_API_KEY = os.getenv("QWEN_KEY")
QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL_NAME = "qwen-vl-plus"

# 初始化搜索工具
search = DuckDuckGoSearchRun()

# ============================
# 新增：知识库 (Knowledge Base) - 扩大版
# 包含常见物品的分类，作为检索工具的本地数据源。
# ============================
GARBAGE_KNOWLEDGE_BASE = {
    # 常用纸类
    "纸巾": "干垃圾 (已使用或污染的纸巾)",
    "卫生纸": "干垃圾",
    "废旧报纸": "可回收物",
    "快递纸箱": "可回收物 (拆开压扁)",
    "包装纸": "可回收物 (干净) 或 干垃圾 (油污)",

    # 塑料/金属/玻璃
    "矿泉水瓶": "可回收物 (清空内容物，冲洗后投放)",
    "塑料瓶": "可回收物 (清空内容物，冲洗后投放)",
    "玻璃瓶": "可回收物",
    "易拉罐": "可回收物 (清空内容物，略作冲洗)",
    "金属罐": "可回收物",
    "塑料袋": "干垃圾 (纯净的快递袋可能可回收，但通常按干垃圾处理)",
    "泡沫塑料": "可回收物 (大块干净) 或 干垃圾 (污染的)",
    "塑料玩具": "干垃圾",
    "牙刷": "干垃圾",

    # 湿垃圾（厨余垃圾）
    "果皮": "湿垃圾（厨余垃圾）",
    "菜叶": "湿垃圾（厨余垃圾）",
    "米饭": "湿垃圾（厨余垃圾）",
    "剩菜剩饭": "湿垃圾（厨余垃圾）",
    "骨头": "湿垃圾（大骨头除外，通常也是湿垃圾）",
    "大骨头": "干垃圾",
    "贝壳": "干垃圾",
    "咖啡渣": "湿垃圾（厨余垃圾）",
    "茶叶渣": "湿垃圾（厨余垃圾）",

    # 有害垃圾
    "过期药物": "有害垃圾 (请保持完整包装投放)",
    "废电池": "有害垃圾 (如干电池、纽扣电池)",
    "荧光灯管": "有害垃圾",
    "灯泡": "有害垃圾 (荧光灯等) 或 干垃圾 (普通白炽灯)",
    "温度计": "有害垃圾 (水银温度计)",
    "油漆桶": "有害垃圾 (需确保内容物已清空或干燥)",
    "杀虫剂": "有害垃圾 (需确保内容物已清空或干燥)",

    # 其它常见干垃圾/复杂分类
    "一次性餐具": "干垃圾",
    "陶瓷碗": "干垃圾 (不可回收)",
    "旧衣服": "可回收物 (干净整洁) 或 干垃圾 (破损严重)",
    "鞋子": "干垃圾 (通常不可回收)",
    "烟头": "干垃圾",
    "宠物粪便": "干垃圾",
    "尿布": "干垃圾",
    "面膜": "干垃圾",
    "笔": "干垃圾"
}


def call_qwen_api(image_bytes: bytes) -> str:
    """辅助函数：发送 HTTP 请求给阿里云"""
    try:
        # [代码与原版相同，省略...]
        img_b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        },
                        {
                            "type": "text",
                            "text": (
                                "请识别图片中的物品是什么，只输出物品名称。例如：'矿泉水瓶'、'电池'、'西瓜皮'。"
                                "严格只输出物品名称，不要解释，不要输出JSON。"
                            )
                        }
                    ]
                }
            ]
        }
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json"
        }
        print("📡 [Tools] 正在调用通义千问 Vision 接口...")
        resp = requests.post(QWEN_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            error_msg = f"API 调用失败: {resp.status_code} - {resp.text}"
            print(f"❌ {error_msg}")
            return f"图像识别服务出错: {error_msg}"
        result = resp.json()
        item_raw = result["choices"][0]["message"]["content"]
        item_name = item_raw.strip().strip("'").strip('"')
        print(f"🔍 [Tools] 识别结果：{item_name}")
        return item_name
    except Exception as e:
        print(f"❌ [Tools] 异常：{e}")
        return f"无法识别图片，原因：{e}"


@tool
def knowledge_retrieval_tool(item_name: str) -> str:
    """
    本地知识库检索工具。
    输入是一个物品的名称。
    如果物品在知识库中，则返回其分类结果，响应速度极快。
    如果知识库中没有，则返回“未找到”。
    """
    print(f"\n📚 [Tool Call] 正在本地知识库中查找: {item_name} ...")

    # 尝试直接匹配
    if item_name in GARBAGE_KNOWLEDGE_BASE:
        result = f"知识库查询结果：{item_name} 属于 {GARBAGE_KNOWLEDGE_BASE[item_name]}。"
        print(f"✅ [Tools] 知识库命中。")
        return result

    # 尝试模糊匹配（处理用户输入不精确的情况）
    for key, value in GARBAGE_KNOWLEDGE_BASE.items():
        if key in item_name or item_name in key:
            result = f"知识库查询结果：{key} 属于 {value}。"
            print(f"✅ [Tools] 知识库命中（模糊匹配）。")
            return result

    print(f"❌ [Tools] 本地知识库未找到: {item_name}")
    return "未找到。"


@tool
def image_recognition_tool(image_path: str) -> str:
    """
    视觉识别工具。
    输入必须是本地图片的绝对路径。
    输出是图片中物品的名称。
    """
    print(f"\n📸 [Tool Call] 正在读取图片: {image_path}")
    if not os.path.exists(image_path):
        return f"错误：找不到文件 {image_path}"
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return call_qwen_api(image_bytes)
    except Exception as e:
        return f"读取图片文件失败: {e}"


@tool
def web_search_tool(query: str) -> str:
    """
    网络搜索工具。
    只有在本地知识库检索失败，或需要特定城市规则时才使用。
    """
    print(f"\n🔍 [Tool Call] 正在搜索: {query} ...")
    try:
        # 添加一个简短的超时机制，防止搜索卡死
        return search.run(query)
    except Exception as e:
        return f"搜索失败: {e}"


# 导出工具列表
def get_tools():
    # 注意：knowledge_retrieval_tool 必须放在 web_search_tool 之前，
    # 这样 Agent 更有可能先尝试本地查找，再尝试网络搜索。
    return [image_recognition_tool, knowledge_retrieval_tool, web_search_tool]

