"""
Author: Zhengyuan Dong
Created: 2025-03-29
Description: Extract and crop images and tables from a PDF file.
Usage: 
python -m src.data_ingestion.tmp_pdf2tabimg --pdf_path downloaded_pdfs/fa97ff8f47026ee384587be81e379b77bfd357bd5c2cecbc646136e627805fe3.pdf --output_folder tmp
"""
import fitz
import os
import io
import pdfplumber
from PIL import Image, ImageOps, ImageChops
import argparse

def auto_crop_image(image, bg_color=(255, 255, 255)):
    """
    Crop the image by detecting the content area.
    """
    # Convert the image to grayscale
    gray_image = image.convert("L")
    
    # Invert the grayscale image
    inverted_image = ImageChops.invert(gray_image)
    
    # Create a bounding box around the non-background area
    bbox = inverted_image.getbbox()
    
    # Crop the image to the bounding box
    if bbox:
        return image.crop(bbox)
    else:
        # Return original if no content is detected
        return image

def extract_and_crop_images_from_pdf(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    image_count = 0

    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Load image with Pillow
            image = Image.open(io.BytesIO(image_bytes))

            # Auto-crop the image based on content
            cropped_image = auto_crop_image(image)

            # Optionally, add a tiny border for readability
            cropped_image = ImageOps.expand(cropped_image, border=2, fill='white')

            # Save the cropped image
            image_filename = f"page_{page_number + 1}_img_{img_index + 1}.png"
            image_path = os.path.join(output_folder, image_filename)
            cropped_image.save(image_path, "PNG", quality=100)
            print(f"Extracted and cropped {image_filename} - saved at {image_path}")
            image_count += 1

    print(f"Total images extracted and cropped: {image_count}")

def save_tables_as_images(pdf_path, output_folder, padding=0, dpi=150):
    os.makedirs(output_folder, exist_ok=True)
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_tables = page.find_tables()
            for table_index, table in enumerate(page_tables):
                x0, top, x1, bottom = map(float, table.bbox)
                # Calculate padding and apply scaling based on DPI
                scale_factor = dpi / 72  # Assuming 72 DPI as default
                padded_bbox = (
                    int(max(x0 - padding, 0) * scale_factor),
                    int(max(top - padding, 0) * scale_factor),
                    int(min(x1 + padding, page.width) * scale_factor),
                    int(min(bottom + padding, page.height) * scale_factor)
                )
                # Render the page as an image with specified DPI
                try:
                    page_image = page.to_image(resolution=dpi).original
                    cropped_image = page_image.crop(padded_bbox)
                    
                    # Automatically trim whitespace around the cropped image tightly
                    cropped_image = ImageOps.crop(cropped_image)
                    # Optionally add a very small border for readability
                    cropped_image = ImageOps.expand(cropped_image, border=2, fill='white')
                    # Save the tightly-cropped image
                    image_filename = f"page_{page_number}_table_{table_index + 1}.png"
                    image_path = os.path.join(output_folder, image_filename)
                    cropped_image.save(image_path)
                    print(f"Extracted table image saved as {image_filename} - saved at {image_path}")
                except Exception as e:
                    print(f"Error processing table on page {page_number}, table {table_index + 1}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract and crop images and tables from a PDF file")
    parser.add_argument("--pdf_path", help="Path to the input PDF file")
    parser.add_argument("--output_folder", help="Output folder to save extracted images and tables")
    parser.add_argument("--padding", type=int, default=0, help="Padding for table extraction (default: 0)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for rendering tables (default: 150)")
    args = parser.parse_args()

    # Define output directories for images and tables
    images_output = os.path.join(args.output_folder, "extracted_images")
    tables_output = os.path.join(args.output_folder, "extracted_table")

    # Process the PDF
    extract_and_crop_images_from_pdf(args.pdf_path, images_output)
    save_tables_as_images(args.pdf_path, tables_output, padding=args.padding, dpi=args.dpi)

if __name__ == "__main__":
    main()