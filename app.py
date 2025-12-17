import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# 导入构建函数
from agent import build_garbage_agent

# 加载环境变量 (虽然 Ollama 不需要 Key，但你的 tools.py 里可能还需要 DuckDuckGo 或其他配置)
load_dotenv()

app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ----------------------------------------------------
# 初始化 Agent (使用本地 Ollama，无需 Key)
# ----------------------------------------------------
print("正在连接本地 Ollama 模型...")
try:
    # 直接调用，不传参
    agent_executor = build_garbage_agent()
    print("✅ 本地 Agent 初始化成功！")
except Exception as e:
    print(f"❌ Agent 初始化失败: {e}")
    print("请检查：\n1. Ollama 软件是否已开启？\n2. 是否运行了 'ollama run qwen3:8b'？")
    agent_executor = None


@app.route('/')
def home():
    # 当访问 http://127.0.0.1:5000/ 时，显示你的 HTML
    return render_template('chat.html')


@app.route('/chat', methods=['POST'])
def chat():
    # 简单的错误保护
    if not agent_executor:
        return jsonify({"status": "error", "reply": "Agent 未成功启动，请检查后台日志。"})

    try:
        # 1. 获取前端发来的数据
        user_text = request.form.get('message', '')
        image_file = request.files.get('image')

        input_prompt = user_text

        # 2. 如果有图片，先保存到本地，再把路径喂给 Agent
        if image_file:
            filename = secure_filename(image_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(file_path)

            # 将路径转为绝对路径给 Agent 读取
            abs_path = os.path.abspath(file_path)
            input_prompt = f"{user_text} (图片路径: {abs_path})"

        print(f"收到请求: {input_prompt}")

        # 3. 调用 Agent
        response = agent_executor.invoke({"input": input_prompt})
        final_answer = response["output"]

        # 4. 返回 JSON 数据给前端
        return jsonify({"status": "success", "reply": final_answer})

    except Exception as e:
        print(f"Error: {e}")
        # 如果模型输出格式乱了，尝试给用户一个友好的提示
        return jsonify({"status": "error", "reply": f"思考过程中出现了一点小问题，请重试。\n({e})"})


if __name__ == '__main__':
    # 启动 Flask
    # 注意: debug=True 导致 Agent 初始化两次，一次是主进程，一次是重载器进程。
    app.run(debug=True, port=5000)

