"""
Author: Zhengyuan Dong
Date: 2025-02-15
Description: A collection of utility classes for extracting and handling data from text.
"""

import re
import os
import pandas as pd
from urllib.parse import urlparse
from io import StringIO

class BibTeXExtractor:
    """Class to extract complete BibTeX entries from a given text."""
    @staticmethod
    def extract(content: str):
        bibtex_entries = []
        bibtex_pattern = r"@(\w+)\{"
        current_entry = ""
        open_braces = 0
        inside_entry = False
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if not inside_entry and re.match(bibtex_pattern, line):
                inside_entry = True
                current_entry = line
                open_braces = line.count("{") - line.count("}")
            elif inside_entry:
                current_entry += " " + line
                open_braces += line.count("{") - line.count("}")

            if inside_entry and open_braces == 0:
                bibtex_entries.append(current_entry.strip())
                inside_entry = False
                current_entry = ""
        return bibtex_entries

class LinkExtractor:
    """Class to extract and validate links from a given text."""
    def __init__(self):
        """Initialize with a list of valid PDF domains."""
        self.VALID_PDF_LINKS = [
            "arxiv.org", "biorxiv.org", "medrxiv.org", "dl.acm.org", "dblp.uni-trier.de", 
            "scholar.google.com", "pubmed.ncbi.nlm.nih.gov", "frontiersin.org", "mdpi.com", 
            "cvpr.thecvf.com", "nips.cc", "icml.cc", "ijcai.org", "webofscience.com", 
            "journals.plos.org", "nature.com", "semanticscholar.org", "chemrxiv.org", 
            "link.springer.com", "ieeexplore.ieee.org", "aaai.org", "openaccess.thecvf.com"
        ]

    def _is_valid_pdf_link(self, link: str) -> bool:
        """Check if a link is a valid PDF link."""
        try:
            parsed = urlparse(link)
            domain = parsed.netloc.lstrip("www.").lower()
        except:
            return False
        for valid_pdf in self.VALID_PDF_LINKS:
            if valid_pdf in link.lower():
                return True
        if link.lower().endswith(".pdf"):
            return True
        return False

    def extract_links(self, text: str):
        """Extract all links, filtering for valid PDF and GitHub links. Clean links to remove unwanted trailing characters."""
        all_links = [lk.strip(".,)|") for lk in re.findall(r"(https?://\S+|www\.\S+)", text)]
        if not all_links:
            return {"pdf_link": None, "github_link": None, "all_links": []}
        # Clean links to remove invalid characters but retain '#' for markdown references
        cleaned_links = [re.sub(r"[)|]+$", "", lk) for lk in all_links]
        pdf_links = [lk for lk in cleaned_links if self._is_valid_pdf_link(lk)]
        github_links = [lk for lk in cleaned_links if "github.com" in lk.lower() or "github.io" in lk.lower()]
        return {
            "pdf_link": pdf_links if pdf_links else None,
            "github_link": github_links if github_links else None,
            "all_links": cleaned_links,
        }

class MarkdownHandler:
    @staticmethod
    def clean_markdown_table(markdown_text):
        """Clean and standardize markdown tables for CSV conversion with bold/italic handling."""
        cleaned_lines = []
        for line in markdown_text.split("\n"):
            # Remove lines like '---' or ':---:'
            if re.match(r"^\s*[:\-|]+\s*$", line):
                continue
            # Remove markdown formatting (bold, italic, inline code)
            line = re.sub(r"(\*\*|\*|`)", "", line)
            # Add missing pipes if the line contains '|' but isn't properly formatted
            if "|" in line and not (line.startswith("|") and line.endswith("|")):
                line = "|" + line.strip() + "|"
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    @staticmethod
    def standardize_table_format(cleaned_markdown):
        """Standardize table format by ensuring each row has the same number of columns."""
        rows = cleaned_markdown.split("\n")
        max_columns = max(len(row.split("|")) - 2 for row in rows if "|" in row)  # Exclude leading and trailing '|'

        standardized_rows = []
        for row in rows:
            columns = row.split("|")[1:-1]  # Remove leading/trailing '|'
            if len(columns) < max_columns:
                columns.extend([""] * (max_columns - len(columns)))  # Fill missing columns
            elif len(columns) > max_columns:
                columns = columns[:max_columns]  # Truncate extra columns
            standardized_rows.append("|" + "|".join(columns) + "|")
        return "\n".join(standardized_rows)

    @staticmethod
    def markdown_to_csv(markdown_text, output_path):
        """Convert cleaned and standardized markdown table to CSV and save it."""
        cleaned_markdown = MarkdownHandler.clean_markdown_table(markdown_text)
        standardized_markdown = MarkdownHandler.standardize_table_format(cleaned_markdown)
        try:
            df = pd.read_csv(StringIO(standardized_markdown), sep="|", engine='python').dropna(axis=1, how="all")
            df.to_csv(output_path, index=False, encoding="utf-8")
            return output_path
        except Exception as e:
            print(f"Failed to convert markdown to CSV for {output_path}: {e}")
            return None

class ExtractionFactory:
    """Factory to create instances of different extractors."""
    @staticmethod
    def get_extractor(extractor_type: str):
        if extractor_type.lower() == "bibtex":
            return BibTeXExtractor
        elif extractor_type.lower() == "link":
            return LinkExtractor()
        elif extractor_type.lower() == "markdown":
            return MarkdownHandler()
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")

if __name__ == "__main__":
    # Example 1: Extract BibTeX
    bibtex_extractor = ExtractionFactory.get_extractor("bibtex")
    sample_bibtex_text = '''
        @article{sample2025,
          title={An amazing discovery},
          author={John Doe},
          journal={Journal of Discoveries},
          year={2025},
          volume={42},
          pages={1-10},
        }
    '''
    print("Extracted BibTeX:", bibtex_extractor.extract(sample_bibtex_text))

    # Example 2: Extract Links
    link_extractor = ExtractionFactory.get_extractor("link")
    sample_markdown = '''
        Check out this [arXiv paper](https://arxiv.org/abs/2201.12345) and this
        [GitHub repo](https://github.com/example/repo). Direct PDF: https://arxiv.org/pdf/2201.12345.pdf
    '''
    print("\nExtracted Links:", link_extractor.extract_links(sample_markdown))

    # Example 3: Convert Markdown to CSV
    markdown_handler = ExtractionFactory.get_extractor("markdown")
    sample_markdown_table = '''
        | Model | #params | Language |
        |-------|---------|----------|
        | bert-base | 110M | English |
    '''
    markdown_handler.markdown_to_csv(sample_markdown_table, "example_markdown.csv")
    print("\nSaved example markdown table as CSV.")