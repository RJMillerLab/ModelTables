import requests

def is_text_file(url):
    """
    通过 HEAD 请求获取 URL 返回的 Content-Type，
    如果类型以 'text/' 开头，则认为是文本文件。
    """
    try:
        # 允许重定向，设置适当的超时时间
        response = requests.head(url, allow_redirects=True, timeout=5)
        content_type = response.headers.get("Content-Type", "")
        # 分割掉可能的 charset 信息，只保留 MIME 类型
        mime_type = content_type.split(";")[0].strip()
        # 判断 MIME 类型是否以 text/ 开头
        return mime_type.startswith("text/")
    except Exception as e:
        print(f"Error checking {url}: {e}")
        return False

urls = [
    "https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json",  # 包含 blob，返回无效
    "https://huggingface.co/DaiShiResearch/maskrcnn-transnext-base-coco/resolve/main/mask_rcnn_transnext_base_fpn_1x_coco_in1k.pth",  # 扩展名 .pth，返回无效
    "https://github.com/batmen-lab/BioMANIA",
    "https://example.com/somefile.md",    # 假设返回 text/markdown 或 text/plain，应返回有效
    "https://example.com/somepage.html"   # 假设返回 text/html，应返回有效
]

for url in urls:
    result = is_text_file(url)
    print(f"{url} -> {result}")
