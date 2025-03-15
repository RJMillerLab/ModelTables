# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-03-13
Description: 增强版标题提取工具，支持缓存、并行处理和断点续传
"""

import time
import os
import json
import re
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import io
import logging
from src.utils import load_config

# 配置日志记录
logging.basicConfig(
    filename='title_extraction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

HEADERS = {"User-Agent": "Mozilla/5.0"}
ARXIV_IGNORE_IDS = ["1910.09700"]
MAX_WORKERS = 8  # 并发线程数
MAX_RETRIES = 3  # 最大重试次数
TIMEOUT = 15     # 请求超时时间

def setup_caching(config):
    """初始化缓存目录和文件路径"""
    cache_dir = os.path.join(config.get('base_path'), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return {
        'arxiv': os.path.join(cache_dir, 'arxiv_cache.json'),
        'url': os.path.join(cache_dir, 'url_cache.json')
    }

def load_cache(cache_path):
    """加载缓存数据"""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"加载缓存失败 {cache_path}: {str(e)}")
    return defaultdict(str)

def save_cache(cache_path, data):
    """保存缓存数据"""
    try:
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logging.error(f"保存缓存失败 {cache_path}: {str(e)}")

def fetch_with_retry(func, item, delay=1):
    """带重试机制的请求包装函数"""
    for attempt in range(MAX_RETRIES):
        try:
            result = func(item)
            if result:  # 如果成功获取结果立即返回
                return result
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed for {item}: {str(e)}")
            time.sleep(delay * (2 ** attempt))  # 指数退避
    return ""

def fetch_arxiv_title(arxiv_id):
    """获取arXiv论文标题（带重试逻辑）"""
    if arxiv_id in ARXIV_IGNORE_IDS:
        return ""
    try:
        response = requests.get(
            f'http://export.arxiv.org/api/query?id_list={arxiv_id}',
            headers=HEADERS,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        if title := root.find('.//{http://www.w3.org/2005/Atom}title'):
            return title.text.strip()
    except Exception as e:
        logging.error(f"arXiv请求失败 {arxiv_id}: {str(e)}")
        raise  # 抛出异常以供重试机制捕获
    return ""

def fetch_url_title(url):
    """获取网页标题（支持PDF和HTML）"""
    try:
        if url.lower().endswith('.pdf'):
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
            with io.BytesIO(response.content) as pdf_file:
                return PdfReader(pdf_file).metadata.get('/Title', '') or ''
        else:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser').title.string.strip()
    except Exception as e:
        logging.error(f"URL请求失败 {url}: {str(e)}")
        raise  # 抛出异常以供重试机制捕获

def batch_fetch(items, fetch_func, cache, cache_path, desc):
    """批量获取数据并更新缓存"""
    new_items = [item for item in items if item not in cache]
    if not new_items:
        logging.info(f"没有需要获取的新条目: {desc}")
        return cache

    logging.info(f"开始获取 {len(new_items)} 条新条目: {desc}")
    progress_bar = tqdm(total=len(new_items), desc=desc)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_with_retry, fetch_func, item): item for item in new_items}
        
        for future in as_completed(futures):
            item = futures[future]
            try:
                cache[item] = future.result()
            except Exception as e:
                logging.error(f"处理失败 {item}: {str(e)}")
                cache[item] = ""
            progress_bar.update(1)
            save_cache(cache_path, cache)  # 每次更新后保存缓存

    progress_bar.close()
    return cache

def main():
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), "processed")
    cache_paths = setup_caching(config)

    # 步骤1: 加载数据
    logging.info("步骤1: 加载数据")
    df = pd.read_parquet(
        os.path.join(processed_base_path, "modelcard_step1.parquet"),
        columns=['parsed_bibtex_tuple_list', 'card_tags', 'all_links']
    )

    # 步骤2: 从BibTeX提取标题
    logging.info("步骤2: 处理BibTeX")
    df["title_list_from_bibtex"] = df["parsed_bibtex_tuple_list"].apply(
        lambda x: [d.get("title", "").replace("{", "").replace("}", "").lower().strip()
        for d in x if isinstance(d, dict)]
    )
    df.to_parquet(os.path.join(processed_base_path, "modelcard_step2_bibtex.parquet"))

    # 步骤3: 处理arXiv
    logging.info("步骤3: 处理arXiv")
    arxiv_cache = load_cache(cache_paths['arxiv'])
    df["arxiv_ids"] = df["card_tags"].apply(
        lambda x: re.findall(r'arxiv[:\s]*([\d\.]+)', x, flags=re.I) if pd.notna(x) else []
    )
    all_arxiv_ids = {aid for sublist in df["arxiv_ids"] for aid in sublist} - set(ARXIV_IGNORE_IDS)
    
    arxiv_cache = batch_fetch(all_arxiv_ids, fetch_arxiv_title, arxiv_cache, 
                            cache_paths['arxiv'], "arXiv标题")
    df["title_list_from_arxiv"] = df["arxiv_ids"].apply(
        lambda ids: [arxiv_cache.get(aid, "") for aid in ids]
    )
    df.to_parquet(os.path.join(processed_base_path, "modelcard_step3_arxiv.parquet"))

    # 步骤4: 处理URL
    logging.info("步骤4: 处理URL")
    url_cache = load_cache(cache_paths['url'])
    df["url_list"] = df["all_links"].apply(
        lambda x: [u.strip() for u in x.split(",")] if pd.notna(x) else []
    )
    all_urls = {url for sublist in df["url_list"] for url in sublist if url}
    
    url_cache = batch_fetch(all_urls, fetch_url_title, url_cache,
                           cache_paths['url'], "URL标题")
    df["title_list_from_url"] = df["url_list"].apply(
        lambda urls: [url_cache.get(url, "") for url in urls]
    )
    df.to_parquet(os.path.join(processed_base_path, "modelcard_step4_url.parquet"))

    # 最终处理
    logging.info("步骤5: 合并结果")
    df["merged_title_list"] = df.apply(
        lambda row: list(set(filter(None, 
            row["title_list_from_bibtex"] +
            row["title_list_from_arxiv"] +
            row["title_list_from_url"]
        ))),
        axis=1
    )
    df.to_parquet(os.path.join(processed_base_path, "modelcard_final.parquet"))
    logging.info("处理完成")

if __name__ == "__main__":
    main()