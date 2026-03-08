#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Zhengyuan Dong
Created: 2025-02-23
Last Modified: 2025-02-24
Description: Extract tables from PDFs, perform OCR, and save results with unique filenames in a unified directory.
"""

import os
import io
import fitz  # PyMuPDF
import pdfplumber  # Extract tables from PDFs
import pandas as pd
from PIL import Image, ImageOps, ImageChops
import pytesseract
import csv
from src.utils import load_config, to_parquet
from tqdm import tqdm
import hashlib

# Paths
CACHE_FILE = "processed_cache.parquet"  # Cache file to avoid reprocessing
TABLES_OUTPUT_FOLDER = "extracted_tables"  # Unified output directory for tables

def load_cache():
    """Load the processed cache if it exists."""
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE).set_index("local_path").to_dict()["csv_paths"]
    return {}

def save_cache(cache_dict):
    """Save the processed cache to a local file."""
    cache_df = pd.DataFrame(list(cache_dict.items()), columns=["local_path", "csv_paths"])
    to_parquet(cache_df, CACHE_FILE)

def auto_crop_image(image):
    """Automatically crop blank margins around the image."""
    gray_image = image.convert("L")
    inverted_image = ImageChops.invert(gray_image)
    bbox = inverted_image.getbbox()
    return image.crop(bbox) if bbox else image

def ocr_image_to_csv(image_path, output_csv_path):
    """Perform OCR on a single table image and save the extracted text as a CSV file."""
    image = Image.open(image_path)
    ocr_result = pytesseract.image_to_string(image, config='--psm 6')
    lines = ocr_result.split('\n')
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for line in lines:
            row = line.split()
            if row:
                csv_writer.writerow(row)

def extract_tables_from_pdf(pdf_path, output_folder, padding=0, dpi=150):
    """Extract table images from a PDF using pdfplumber and save them in a unified folder."""
    os.makedirs(output_folder, exist_ok=True)
    table_images = []
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]  # Unique identifier for the PDF

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                tables = page.find_tables()
                if not tables:
                    continue  # Skip pages without tables

                page_image = page.to_image(resolution=dpi).original
                for table_idx, table in enumerate(tables):
                    x0, top, x1, bottom = map(float, table.bbox)
                    scale_factor = dpi / 72  # Convert coordinates based on DPI

                    # Apply padding and scale coordinates
                    padded_bbox = (
                        int(max(x0 - padding, 0) * scale_factor),
                        int(max(top - padding, 0) * scale_factor),
                        int(min(x1 + padding, page.width) * scale_factor),
                        int(min(bottom + padding, page.height) * scale_factor)
                    )

                    try:
                        cropped_image = page_image.crop(padded_bbox)
                        cropped_image = ImageOps.crop(cropped_image)  # Auto-trim whitespace
                        cropped_image = ImageOps.expand(cropped_image, border=2, fill='white')

                        # Save the table image with a unique filename
                        image_filename = f"{pdf_basename}_page{page_number}_table{table_idx + 1}.png"
                        image_path = os.path.join(output_folder, image_filename)
                        cropped_image.save(image_path)
                        table_images.append(image_path)
                    except Exception as e:
                        print(f"⚠️ Error processing table on {pdf_basename}, page {page_number}, table {table_idx + 1}: {e}")

    except Exception as e:
        print(f"❌ Skipping corrupt or unreadable PDF: {pdf_path} | Error: {e}")
        return []  # Return empty list so script continues without crashing

    return table_images


def extract_and_ocr_pdfs(df_unique, base_dir, cache_dict):
    """
    Process only PDFs that contain tables.
    1. Extract table images from PDFs.
    2. Perform OCR on the extracted table images.
    3. Save OCR results as CSV and update the cache.
    """
    new_results = {}
    tables_output_path = os.path.join(base_dir, TABLES_OUTPUT_FOLDER)
    os.makedirs(tables_output_path, exist_ok=True)  # Ensure unified folder exists

    for idx, row in tqdm(df_unique.iterrows(), total=len(df_unique), desc="Processing PDFs with Tables"):
        pdf_path = row.get('local_path', None)

        # Skip missing or unreadable files
        if not pdf_path or not os.path.isfile(pdf_path) or os.path.getsize(pdf_path) < 1000:
            print(f"⚠️ Skipping invalid or corrupted file: {pdf_path}")
            new_results[pdf_path] = []
            continue
        
        # Skip PDFs already processed in cache
        if pdf_path in cache_dict:
            new_results[pdf_path] = cache_dict[pdf_path]
            continue

        # Extract table images from the PDF
        table_images = extract_tables_from_pdf(pdf_path, tables_output_path)
        if not table_images:
            new_results[pdf_path] = []
            cache_dict[pdf_path] = []
            continue  # Skip if no tables were found

        csv_file_list = []
        for table_img_path in table_images:
            csv_name = os.path.splitext(os.path.basename(table_img_path))[0] + ".csv"
            csv_path = os.path.join(tables_output_path, csv_name)
            ocr_image_to_csv(table_img_path, csv_path)
            csv_file_list.append(csv_path)

        # Update the cache
        new_results[pdf_path] = csv_file_list
        cache_dict[pdf_path] = csv_file_list  # Directly write to cache

    return new_results


def main_process():
    """
    1. Load the downloaded PDF information.
    2. Load the cache and skip already processed table images.
    3. Process only new PDFs that contain tables.
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

    # Process only PDFs that contain tables and are not cached
    processed_results = extract_and_ocr_pdfs(df_unique, base_path, cache_dict)

    # Map the OCR results back to the original DataFrame
    df_info['csv_paths'] = df_info['local_path'].map(lambda x: cache_dict.get(x, []))

    # Save the updated cache
    save_cache(cache_dict)

    # Save the processed DataFrame as a new Parquet file
    output_parquet = os.path.join(base_path, "processed", "processed_pdf_with_ocr.parquet")
    to_parquet(df_info, output_parquet)
    print(f"✅ Processing complete! Added column 'csv_paths' and saved to {output_parquet}")

if __name__ == "__main__":
    main_process()
