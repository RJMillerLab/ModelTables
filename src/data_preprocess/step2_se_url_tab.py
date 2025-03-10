# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-02-27
Description: Extract title from bibtex，同时新增从 arXiv id 和 URL 中提取论文标题
"""
import time, os, json, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from src.utils import load_config

# 新增的依赖
import requests  ########
from xml.etree import ElementTree as ET  ########
from bs4 import BeautifulSoup  ########
import re  ########
from PyPDF2 import PdfReader  ########
import io  ########

ARXIV_IGNORE_IDS = ["1910.09700"]  ########

def extract_titles(bibtex_list):
    if not isinstance(bibtex_list, (list, tuple, np.ndarray)):
        return []
    return [d.get("title", "").replace("{", "").replace("}", "").lower().strip()
            for d in bibtex_list if isinstance(d, dict) and d.get("title")]  ########

def extract_arxiv_ids(tag_str):
    if not tag_str:
        return []
    return re.findall(r'arxiv[:\s]*([\w\.-]+)', tag_str, flags=re.IGNORECASE)  ########

def get_arxiv_title(arxiv_id):
    if arxiv_id in ARXIV_IGNORE_IDS:
        return ""
    api_url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'  ########
    try:
        response = requests.get(api_url, timeout=10)  ########
        if response.status_code == 200:
            root = ET.fromstring(response.text)  ########
            # arXiv API 返回的 XML 中 <entry> 内含有 <title> 标签
            entry = root.find('{http://www.w3.org/2005/Atom}entry')  ########
            if entry is not None:
                title_elem = entry.find('{http://www.w3.org/2005/Atom}title')  ########
                if title_elem is not None:
                    return title_elem.text.strip()  ########
    except Exception as e:
        return ""
    return ""

def get_url_title(url):
    if url.lower().endswith('.pdf'):
        try:
            r = requests.get(url, timeout=10)  ########
            if r.status_code == 200:
                pdf_file = io.BytesIO(r.content)  ########
                reader = PdfReader(pdf_file)  ########
                meta = reader.metadata
                title = meta.title if meta and meta.title else ""
                print(f"PDF URL: {url} -> Title: {title}")  ########
                return title.strip() if title else ""
        except Exception as e:
            print(f"Error extracting PDF title for {url}: {e}")  ########
            return ""
    try:
        resp = requests.get(url, timeout=10)  ########
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')  ########
            title_tag = soup.find('title')  ########
            if title_tag:
                extracted_title = title_tag.get_text().strip()  ########
                print(f"HTML URL: {url} -> Title: {extracted_title}")  ########
                return extracted_title
    except Exception as e:
        print(f"Error extracting HTML title for {url}: {e}")  ########
        return ""
    return ""

def main():
    data_type = "modelcard"
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), "processed")
    
    print("⚠️ Step 1: Loading data...")
    start_time = time.time()
    df = pd.read_parquet(os.path.join(processed_base_path, f"{data_type}_step1.parquet"), 
                         columns=['parsed_bibtex_tuple_list', 'card_tags', 'all_links'])  ########
    print(f"✅ Data loaded. Time cost: {time.time() - start_time:.2f} seconds.")
    
    print("⚠️ Step 2: Extracting titles from BibTeX...")
    start_time = time.time()
    tqdm.pandas()  ########
    df["title_list_from_bibtex"] = df["parsed_bibtex_tuple_list"].progress_apply(extract_titles)  ########
    print(f"✅ BibTeX titles extracted. Time cost: {time.time() - start_time:.2f} seconds.")
    
    print("⚠️ Step 3: Building title to paper index mapping...")
    start_time = time.time()
    title_to_paper_indices = defaultdict(set)
    for idx, titles in df["title_list_from_bibtex"].items():
        for title in titles:
            title_to_paper_indices[title].add(idx)
    print(f"✅ Title mapping built. Time cost: {time.time() - start_time:.2f} seconds.")
    
    print("⚠️ Step 4: Extracting titles from arXiv and URL sources...")
    start_time = time.time()
    #df["title_list_from_arxiv"] = df["card_tags"].apply(
    #    lambda x: [get_arxiv_title(arxiv_id) for arxiv_id in extract_arxiv_ids(x) if arxiv_id not in ARXIV_IGNORE_IDS] if pd.notna(x) else [] )  ########
    df["title_list_from_arxiv"] = df["card_tags"].progress_apply(
        lambda x: [get_arxiv_title(arxiv_id) for arxiv_id in extract_arxiv_ids(x) if arxiv_id not in ARXIV_IGNORE_IDS] if pd.notna(x) else [] )
    df["title_list_from_url"] = df["all_links"].progress_apply(
        lambda links: [get_url_title(link) for link in links.split(", ") if link] if pd.notna(links) else [] )  ########
    print(f"✅ arXiv and URL titles extracted. Time cost: {time.time() - start_time:.2f} seconds.")
    
    print("⚠️ Step 5: Merging all title lists...")
    df["merged_title_list"] = df.apply(
        lambda row: list(set(row["title_list_from_bibtex"] + row["title_list_from_arxiv"] + row["title_list_from_url"])),
        axis=1)  ########
    print("✅ Title lists merged.")
    
    print("⚠️ Step 6: Saving results...")
    output_path = os.path.join(config.get('base_path'), "processed", f"{data_type}_with_titles.parquet")  ########
    df.to_parquet(output_path)  ########
    print(f"✅ Results saved to {output_path}")

if __name__ == "__main__":
    main()
