#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Zhengyuan Dong
Created: 2025-02-23
Last Modified: 2025-02-23
Description: Extract tables from PDF files and save to CSV files.
"""

import os
import re
import pandas as pd
import fitz
from tqdm import tqdm
from joblib import Parallel, delayed
from pdf2image import convert_from_path
import cv2
import pytesseract
import numpy as np
from src.utils import load_config

def pdf_to_images(pdf_path, dpi=300):
    images = convert_from_path(pdf_path, dpi=dpi)
    return images

def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return binary

def extract_table_from_image(image):
    custom_config = "--psm 6"
    extracted_text = pytesseract.image_to_string(image, config=custom_config)
    return extracted_text

def extract_table_from_pdf(pdf_path):
    images = pdf_to_images(pdf_path)
    table_data = []
    for img in images:
        processed_img = preprocess_image(img)
        table_text = extract_table_from_image(processed_img)
        lines = table_text.split('\n')
        for line in lines:
            row = re.split(r'\s{2,}', line.strip())
            if len(row) > 1:
                table_data.append(row)
    return table_data if table_data else None

def process_pdf_row(row, output_folder = "pdf_extracted_csv"):
    os.makedirs(output_folder, exist_ok=True)
    local_path = row.get('local_path')
    if pd.notnull(local_path) and os.path.isfile(local_path):
        table = extract_table_from_pdf(local_path)
        if table:
            csv_filename = os.path.splitext(os.path.basename(local_path))[0] + ".csv"
            output_csv_path = os.path.join(output_folder, csv_filename)
            try:
                pd.DataFrame(table).to_csv(output_csv_path, index=False, header=False, encoding="utf8")
            except Exception as e:
                print(f"Error saving CSV for {local_path}: {e}")
        return table
    else:
        return None

def parallel_process_pdf_entries(df):
    results = Parallel(n_jobs=-1)(
        delayed(process_pdf_row)(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing PDFs")
    )
    df = df.copy()
    df['extracted_pdf_table'] = results
    return df

def main():
    config = load_config('config.yaml')
    base_path = config.get('base_path')
    pdf_csv_path = os.path.join(base_path, "downloaded_pdfs_by_model_info.parquet")
    
    print("Loading CSV files...")
    df_pdf = pd.read_parquet(pdf_csv_path)
    df_pdf = df_pdf[df_pdf['local_path'].notnull()].copy()
    print("Extracting tables from PDF files in parallel...")
    df_pdf = parallel_process_pdf_entries(df_pdf)
    print("Saving extracted tables to CSV files...")
    output_parquet = os.path.join(base_path, "step_pdf_table.parquet")
    df_pdf.to_parquet(output_parquet, index=False)
    print("Final data saved as Parquet file:", output_parquet)

if __name__ == "__main__":
    main()
