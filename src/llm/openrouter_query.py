"""
Author: Zhengyuan Dong
Date: 2025-10-27
Description: Simulate batch querying multiple LLMs (Claude, Gemini, DeepSeek, Qwen)
             via OpenRouter unified endpoint with one API key.
"""

import os, json, asyncio, aiohttp, time
from dotenv import load_dotenv

# ==========================================================
# 1️⃣  Configuration
# ==========================================================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = [
    "anthropic/claude-3.5-sonnet",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3-70b-instruct",
    "openai/gpt-3.5-turbo",  # Reliable fallback
    "openai/gpt-4o-mini",
]

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("Please set OPENROUTER_API_KEY in your .env file")


# ==========================================================
# 2️⃣  Helper functions
# ==========================================================
async def query_model(session, model, prompt, max_tokens=1000, temperature=0.2):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
        if resp.status != 200:
            text = await resp.text()
            return {"model": model, "error": f"HTTP {resp.status}: {text}"}
        data = await resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"model": model, "response": content}


async def query_all_models(prompt, session):
    tasks = [query_model(session, m, prompt) for m in MODELS]
    results = await asyncio.gather(*tasks)
    return results


# ==========================================================
# 3️⃣  Batch runner
# ==========================================================
async def batch_query(input_path: str, output_path: str):
    start = time.time()
    with open(input_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line.strip()) for line in f if line.strip()]

    results = []
    async with aiohttp.ClientSession() as session:
        for i, sample in enumerate(samples):
            prompt = sample.get("prompt") or sample.get("text") or sample.get("input")
            if not prompt:
                continue
            print(f"\n🚀 Processing sample {i+1}/{len(samples)}...")
            batch_results = await query_all_models(prompt, session)
            results.append({"sample_id": i, "prompt": prompt, "outputs": batch_results})

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ Completed batch query. Saved to {output_path}")
    print(f"⏱️ Total time: {time.time() - start:.1f}s")


# ==========================================================
# 4️⃣  Entrypoint
# ==========================================================
if __name__ == "__main__":
    input_path = "batch_input_openrouter.jsonl"
    output_path = "batch_output_openrouter.jsonl"
    asyncio.run(batch_query(input_path, output_path))

