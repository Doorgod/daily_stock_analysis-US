import os
import requests
from openai import OpenAI

# 填入你刚才提供的信息
BASE_URL = "https://key.002836.xyz/v1"
API_KEY = "sk-KVadFv9uN2ZSA9QV3o2k1jsFErOSxMLEKYc2W6RujGfsuKTl"
MODEL = "gemini-3-pro-preview"
TG_TOKEN = "7828472094:AAEJQPcbfyJaBBDcHwqfd9dFGc1TiBXooFo"
TG_CHAT_ID = "7084617453"

def test_connection():
    print("开始测试...")
    
    # 1. 测试 AI 接口
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "你好，请回复'AI已就绪'"}]
        )
        print(f"AI 回复: {response.choices[0].message.content}")
    except Exception as e:
        print(f"AI 测试失败: {e}")

    # 2. 测试 Telegram 机器人
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = {"chat_id": TG_CHAT_ID, "text": "🤖 股票机器人本地测试成功！"}
        res = requests.post(url, data=data)
        if res.status_code == 200:
            print("Telegram 消息已发出，请检查手机！")
        else:
            print(f"Telegram 发送失败: {res.text}")
    except Exception as e:
        print(f"Telegram 测试失败: {e}")

if __name__ == "__main__":
    test_connection()