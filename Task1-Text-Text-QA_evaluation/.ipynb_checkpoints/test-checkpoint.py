from openai import OpenAI

client = OpenAI(
    api_key="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",  # 用你的 key 替换
    base_url="https://openrouter.ai/api/v1",
)

try:
    resp = client.chat.completions.create(
        model="mistralai/ministral-3b",  # 请换成你在 OpenRouter 控制台可以看到的真实模型 ID
        messages=[
            {"role": "user", "content": "Hello, this is a billing test."}
        ],
        max_tokens=20,
    )
    print("✅ 调用成功")
    print("模型回复:", resp.choices[0].message.content)
except Exception as e:
    print("❌ 调用失败:", e)