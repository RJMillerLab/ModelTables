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
        
        # Find rows with pipes and calculate max columns
        pipe_rows = [row for row in rows if "|" in row]
        if not pipe_rows:
            # If no rows contain pipes, return the original content
            return cleaned_markdown
        
        # Calculate max columns based on ALL pipe rows, not just header
        # This ensures we capture the maximum number of columns across all rows
        column_counts = [len(row.split("|")) - 2 for row in pipe_rows]  # Exclude leading and trailing '|'
        max_columns = max(column_counts) if column_counts else 0
        
        # If max_columns is 0 or negative, return original content
        if max_columns <= 0:
            return cleaned_markdown

        standardized_rows = []
        for row in rows:
            if "|" in row:
                columns = row.split("|")[1:-1]  # Remove leading/trailing '|'
                if len(columns) < max_columns:
                    columns.extend([""] * (max_columns - len(columns)))  # Fill missing columns
                elif len(columns) > max_columns:
                    columns = columns[:max_columns]  # Truncate extra columns
                standardized_rows.append("|" + "|".join(columns) + "|")
            else:
                # Keep non-pipe rows as-is
                standardized_rows.append(row)
        return "\n".join(standardized_rows)

    @staticmethod
    def detect_table_type(df):
        """Detect if this is a label scheme table or performance table."""
        component_col = None
        labels_col = None
        
        # Look for Component and Labels columns (handle spaces)
        for col in df.columns:
            if 'Component' in col.strip():
                component_col = col
            if 'Labels' in col.strip():
                labels_col = col
        
        return component_col, labels_col

    @staticmethod
    def process_label_scheme_table(df, component_col, labels_col):
        """Process label scheme table by converting comma-separated labels to semicolon-separated."""
        result = []
        for i, row in df.iterrows():
            component = row[component_col]
            labels = row[labels_col]
            if pd.notna(labels):
                labels_str = str(labels)
                
                # Handle both comma and pipe separators
                # First split by comma, then by pipe within each comma-separated group
                label_list = []
                for comma_group in labels_str.split(','):
                    comma_group = comma_group.strip()
                    if comma_group:
                        # Split by pipe and add each part as a separate label
                        pipe_labels = [label.strip() for label in comma_group.split('|') if label.strip()]
                        label_list.extend(pipe_labels)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_labels = []
                for label in label_list:
                    if label not in seen:
                        seen.add(label)
                        unique_labels.append(label)
                
                result.append([component, ';'.join(unique_labels)])
        return result

    @staticmethod
    def markdown_to_csv(markdown_text, output_path, verbose=False, preserve_empty_cols=False):
        """Enhanced convert markdown table to CSV with smart table type detection."""
        cleaned_markdown = MarkdownHandler.clean_markdown_table(markdown_text)
        standardized_markdown = MarkdownHandler.standardize_table_format(cleaned_markdown)
        
        try:
            # Parse the markdown table first
            df = pd.read_csv(StringIO(standardized_markdown), sep="|", engine='python', keep_default_na=False, na_filter=False)
            
            # Check if this is a Label Scheme table (Component + Labels columns)
            is_label_scheme = False
            component_col = None
            labels_col = None
            
            for col in df.columns:
                if 'Component' in col.strip():
                    component_col = col
                if 'Labels' in col.strip():
                    labels_col = col
            
            if component_col and labels_col:
                is_label_scheme = True
                if verbose:
                    print(f"🎯 Detected Label Scheme table: {os.path.basename(output_path)}")
                
                # Special processing for Label Scheme tables
                # Handle pipes in Labels column by merging extra columns
                if len(df.columns) > 2:  # More than Component and Labels
                    # Find the Labels column index
                    labels_idx = df.columns.get_loc(labels_col)
                    
                    # Merge all columns after Labels into Labels column
                    for i, row in df.iterrows():
                        labels_value = str(row[labels_col]) if pd.notna(row[labels_col]) else ''
                        
                        # Collect all values from columns after Labels
                        extra_values = []
                        for j in range(labels_idx + 1, len(df.columns)):
                            if pd.notna(row.iloc[j]) and str(row.iloc[j]).strip():
                                extra_values.append(str(row.iloc[j]).strip())
                        
                        # Merge with Labels column using |
                        if extra_values:
                            labels_value = labels_value + ' | ' + ' | '.join(extra_values)
                            df.at[i, labels_col] = labels_value
                    
                    # Keep only Component and Labels columns
                    df = df[[component_col, labels_col]]
                
                # Process label scheme table by converting comma-separated labels to semicolon-separated
                processed_data = MarkdownHandler.process_label_scheme_table(df, component_col, labels_col)
                result_df = pd.DataFrame(processed_data, columns=['Component', 'Labels'])
            else:
                if verbose:
                    print(f"📈 Detected Performance table: {os.path.basename(output_path)}")
                # Process as performance table (keep as-is)
                result_df = df
            
            # Use our new cleaning logic for consistency
            from src.utils import clean_dataframe_for_analysis
            result_df = clean_dataframe_for_analysis(
                result_df, 
                drop_empty_rows=True, 
                drop_empty_cols=not preserve_empty_cols, 
                preserve_empty_cells=True
            )

            # Check if the DataFrame is empty or has no data rows (only headers)
            if result_df.empty or len(result_df) == 0:
                if verbose:
                    print(f"⚠️ Empty table detected: {os.path.basename(output_path)}")
                return None
            
            # Save the processed data
            result_df.to_csv(output_path, index=False, encoding="utf-8", na_rep='')
            return output_path
            
        except Exception as e:
            if verbose:
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