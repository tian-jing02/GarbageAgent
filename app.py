from dotenv import load_dotenv
import os
# import dashscope
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from agent import build_garbage_agent

load_dotenv()

# ----------------------------------------------------
# 2. 密钥加载和检查 (现在使用 HUGGINGFACE_API_KEY)
# ----------------------------------------------------
# 注意：移除了末尾的逗号，确保它是一个字符串
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

if not hf_api_key:
    # 打印错误并阻止程序运行
    print("❌ 错误：HUGGINGFACE_API_KEY 未找到！请检查 .env 文件。")
    print("请确保 .env 文件在项目根目录，且格式为 HUGGINGFACE_API_KEY=\"hf_...\"")
    exit(1)

app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 初始化 Agent (全局加载一次，提高响应速度)
print("正在初始化 Agent...")
# 传递 Hugging Face 密钥
agent_executor = build_garbage_agent(hf_api_key)


@app.route('/')
def home():
    # 当访问 http://127.0.0.1:5000/ 时，显示你的 HTML
    return render_template('one.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        # 1. 获取前端发来的数据
        user_text = request.form.get('message', '')
        image_file = request.files.get('image')

        input_prompt = user_text

        # 2. 如果有图片，先保存到本地，再把路径喂给 Agent
        if image_file:
            filename = secure_filename(image_file.filename)
            # 为了防止重名，实际开发通常加时间戳，这里简单处理
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(file_path)

            # 将路径转为绝对路径或相对路径给 Agent 读取
            # 注意：Windows下路径分隔符的处理
            abs_path = os.path.abspath(file_path)
            input_prompt = f"{user_text} (图片路径: {abs_path})"

        print(f"收到请求: {input_prompt}")

        # 3. 调用 Agent
        response = agent_executor.invoke({"input": input_prompt})
        final_answer = response["output"]

        # 4. 返回 JSON 数据给前端
        return jsonify({"status": "success", "reply": final_answer})

    except Exception as e:
        # 这里会捕获到模型调用失败（如密钥无效）的错误
        print(f"Error: {e}")
        return jsonify({"status": "error", "reply": f"抱歉，处理时出错了。错误信息：{e}"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)