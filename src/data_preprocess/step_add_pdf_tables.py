#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Zhengyuan Dong
Created: 2025-02-23
Last Modified: 2025-02-24
Description: Extract tables from PDF files and save to CSV files with persistent caching.
"""

import os
import io
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image, ImageOps, ImageChops
import pytesseract
import csv
from src.utils import load_config
from tqdm import tqdm
import hashlib

# Path to the persistent cache file
CACHE_FILE = "processed_cache.parquet"

def load_cache():
    """Load the processed cache if it exists."""
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE).set_index("local_path").to_dict()["csv_paths"]
    return {}

def save_cache(cache_dict):
    """Save the processed cache to a local file."""
    cache_df = pd.DataFrame(list(cache_dict.items()), columns=["local_path", "csv_paths"])
    cache_df.to_parquet(CACHE_FILE, index=False)

def get_cache_key(pdf_path):
    """Generate a unique cache key for a given PDF path."""
    return hashlib.md5(pdf_path.encode()).hexdigest()

def auto_crop_image(image):
    """Automatically crop the blank margins around the image."""
    gray_image = image.convert("L")
    inverted_image = ImageChops.invert(gray_image)
    bbox = inverted_image.getbbox()
    return image.crop(bbox) if bbox else image

def ocr_image_to_csv(image_path, output_csv_path):
    """Perform OCR on a single image and save the extracted text as a CSV file."""
    image = Image.open(image_path)
    ocr_result = pytesseract.image_to_string(image, config='--psm 6')
    lines = ocr_result.split('\n')
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for line in lines:
            row = line.split()
            if row:
                csv_writer.writerow(row)

def extract_and_ocr_pdfs(df_unique, base_dir, cache_dict):
    """
    Process only PDFs that have not been cached.
    1. Extract images from PDFs.
    2. Perform OCR on the extracted images.
    3. Save OCR results as CSV and update the cache.
    """
    new_results = {}

    for idx, row in tqdm(df_unique.iterrows(), total=len(df_unique), desc="Processing Unique PDFs"):
        pdf_path = row.get('local_path', None)
        if not pdf_path or not os.path.isfile(pdf_path):
            new_results[pdf_path] = []
            continue
        
        # Skip processing if the PDF has already been cached
        if pdf_path in cache_dict:
            new_results[pdf_path] = cache_dict[pdf_path]
            continue

        # Process the PDF and extract images
        folder_to_save = os.path.dirname(pdf_path)
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]

        pdf_document = fitz.open(pdf_path)
        csv_file_list = []
        image_counter = 0

        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                image = Image.open(io.BytesIO(image_bytes))
                if image.mode == "CMYK":
                    image = image.convert("RGB")

                cropped_image = auto_crop_image(image)
                cropped_image = ImageOps.expand(cropped_image, border=2, fill='white')

                tmp_img_name = f"{pdf_basename}_p{page_number+1}_img{img_index+1}.png"
                tmp_img_path = os.path.join(folder_to_save, tmp_img_name)
                cropped_image.save(tmp_img_path, "PNG", quality=100)

                image_counter += 1
                csv_name = f"{pdf_basename}_{image_counter}.csv"
                csv_path = os.path.join(folder_to_save, csv_name)
                ocr_image_to_csv(tmp_img_path, csv_path)
                csv_file_list.append(csv_path)

        pdf_document.close()

        # Update the cache
        new_results[pdf_path] = csv_file_list
        cache_dict[pdf_path] = csv_file_list  # Directly write to cache

    return new_results

def main_process():
    """
    1. Load the downloaded PDF information.
    2. Load the cache and skip already processed files.
    3. Process only new PDFs (that were not cached).
    4. Update the cache and save the processed information.
    """
    config = load_config('config.yaml')
    base_path = config.get('base_path')
    
    # Read the downloaded PDFs information
    downloaded_info_path = os.path.join(base_path, "processed", "downloaded_pdfs_info.parquet")
    df_info = pd.read_parquet(downloaded_info_path)

    # Load the persistent cache
    cache_dict = load_cache()

    # Remove duplicates to process each unique PDF only once
    df_unique = df_info[['local_path']].drop_duplicates().reset_index(drop=True)

    # Process only PDFs that are not in the cache
    processed_results = extract_and_ocr_pdfs(df_unique, base_path, cache_dict)

    # Map the OCR results back to the original DataFrame
    df_info['csv_paths'] = df_info['local_path'].map(lambda x: cache_dict.get(x, []))

    # Save the updated cache
    save_cache(cache_dict)

    # Save the processed DataFrame as a new Parquet file
    output_parquet = os.path.join(base_path, "processed", "processed_pdf_with_ocr.parquet")
    df_info.to_parquet(output_parquet, index=False)
    print(f"✅ Processing complete! Added column 'csv_paths' and saved to {output_parquet}")

if __name__ == "__main__":
    main_process()
