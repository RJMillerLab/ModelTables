"""
Author: Zhengyuan Dong
Date: 2025-02-15
Description: A simple BibTeX parser that extracts metadata from BibTeX entries.
"""
# Import necessary modules
from pybtex.database import parse_string
import bibtexparser
import re

# Ensure the BibTeX entry is a decoded string
def ensure_string(bibtex_entry):
    if isinstance(bibtex_entry, bytes):
        try:
            return bibtex_entry.decode("utf-8")
        except UnicodeDecodeError:
            #print("Error decoding BibTeX entry")
            return None
    return bibtex_entry

class BibTeXFactory:
    def __init__(self):
        pass
    @staticmethod
    def parse_bibtex(bibtex_entry, verbose=False):
        """Attempt to parse a BibTeX entry using multiple approaches with a cascade of fallback methods."""
        result = {}
        bibtex_entry = ensure_string(bibtex_entry)
        # First attempt: pybtex
        try:
            entry = parse_string(bibtex_entry, "bibtex")
            for key, entry_data in entry.entries.items():
                result["entry_type"] = entry_data.type
                result["title"] = entry_data.fields.get("title", "")
                result["author"] = entry_data.fields.get("author", "")
                result["year"] = entry_data.fields.get("year", "")
                result["doi"] = entry_data.fields.get("doi", "")
                result["url"] = entry_data.fields.get("url", "")
                result["journal"] = entry_data.fields.get("journal", "")
                result["booktitle"] = entry_data.fields.get("booktitle", "")
                result["volume"] = entry_data.fields.get("volume", "")
                result["pages"] = entry_data.fields.get("pages", "")
            return result, True
        except Exception as e:
            #print(f"pybtex failed: {e}")
            # Second attempt: bibtexparser
            try:
                bib_database = bibtexparser.loads(bibtex_entry)
                if len(bib_database.entries) > 0:
                    entry_data = bib_database.entries[0]
                    result["entry_type"] = entry_data.get("ENTRYTYPE", "")
                    result["title"] = entry_data.get("title", "")
                    result["author"] = entry_data.get("author", "")
                    result["year"] = entry_data.get("year", "")
                    result["doi"] = entry_data.get("doi", "")
                    result["url"] = entry_data.get("url", "")
                    result["journal"] = entry_data.get("journal", "")
                    result["booktitle"] = entry_data.get("booktitle", "")
                    result["volume"] = entry_data.get("volume", "")
                    result["pages"] = entry_data.get("pages", "")
                    return result, True
            except Exception as e:
                #if verbose:
                #    print(f"bibtexparser also failed: {e}")
                # Third attempt: Apply formatting fixes and retry with pybtex
                try:
                    fixed_entry = BibTeXFactory.fix_bibtex_entry(bibtex_entry)
                    entry = parse_string(fixed_entry, "bibtex")
                    for key, entry_data in entry.entries.items():
                        result["entry_type"] = entry_data.type
                        result["title"] = entry_data.fields.get("title", "")
                        result["author"] = entry_data.fields.get("author", "")
                        result["year"] = entry_data.fields.get("year", "")
                        result["doi"] = entry_data.fields.get("doi", "")
                        result["url"] = entry_data.fields.get("url", "")
                        result["journal"] = entry_data.fields.get("journal", "")
                        result["booktitle"] = entry_data.fields.get("booktitle", "")
                        result["volume"] = entry_data.fields.get("volume", "")
                        result["pages"] = entry_data.fields.get("pages", "")
                    return result, True
                except Exception as e:
                    if verbose:
                        print(f"Final parsing attempt after fix failed: {e}")
        # If all methods fail, return None
        return None, False  
    @staticmethod
    def fix_bibtex_entry(entry):
        """Fix common BibTeX issues: missing commas, undefined months, and incorrect author formatting."""
        entry = BibTeXFactory.fix_bibtex_missing_comma(entry)
        entry = BibTeXFactory.fix_bibtex_undefined_month(entry)
        entry = BibTeXFactory.fix_bibtex_author_format(entry)
        return entry
    @staticmethod
    def fix_bibtex_missing_comma(entry):
        """Fix missing commas between BibTeX fields."""
        if not isinstance(entry, str):
            return entry
        entry = re.sub(r'(\w+)\s*=\s*{([^}]*)}\s+(\w+)\s*=', r'\1 = {\2}, \3 =', entry)
        return entry
    @staticmethod
    def fix_bibtex_undefined_month(entry):
        """Convert short month names to full BibTeX format."""
        if not isinstance(entry, str):
            return entry
        months = {
            "jan": "January", "feb": "February", "mar": "March", "apr": "April",
            "may": "May", "jun": "June", "jul": "July", "aug": "August",
            "sep": "September", "oct": "October", "nov": "November", "dec": "December"
        }
        for short, full in months.items():
            entry = re.sub(rf'month\s*=\s*{short}\b', f'month = {{{full}}}', entry, flags=re.IGNORECASE)
        return entry
    @staticmethod
    def fix_bibtex_author_format(entry):
        """Ensure author names are formatted correctly using 'and' instead of commas."""
        if not isinstance(entry, str):
            return entry
        match = re.search(r'author\s*=\s*{(.*?)}', entry, re.DOTALL)
        if match:
            authors = match.group(1)
            fixed_authors = ' and '.join([name.strip() for name in authors.split(',')])
            entry = re.sub(r'author\s*=\s*{.*?}', f'author = {{{fixed_authors}}}', entry, flags=re.DOTALL)
        return entry
    @staticmethod
    def parse_multiple_bibtex_entries(bibtex_entries):
        """Parse multiple BibTeX entries and return a list of structured metadata."""
        parsed_entries = []
        for bibtex_entry in bibtex_entries:
            parsed_entry, _ = BibTeXFactory.parse_bibtex(bibtex_entry)
            if parsed_entry:
                parsed_entries.append(parsed_entry)
        return parsed_entries

if __name__ == "__main__":
    bibtex_entries = bibtex_entries = [
    """
    @inproceedings{camacho-collados-etal-2022-tweetnlp,
        title = "{T}weet{NLP}: Cutting-Edge Natural Language Processing for Social Media",
        author = "Camacho-collados, Jose  and Rezaee, Kiamehr  and Riahi, Talayeh  and Ushio, Asahi  and Loureiro, Daniel  and Antypas, Dimosthenis  and Boisson, Joanne  and Espinosa Anke, Luis  and Liu, Fangyu  and Mart{\\\'\\i}nez C{\\\'a}mara, Eugenio and others",
        booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
        month = dec,
        year = "2022",
        address = "Abu Dhabi, UAE",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2022.emnlp-demos.5",
        pages = "38--49"
    }
    """,
    """
    @article{solatorio2024gistembed,
        title={GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning},
        author={Aivin V. Solatorio},
        journal={arXiv preprint arXiv:2402.16829},
        year={2024},
        URL={https://arxiv.org/abs/2402.16829}
        eprint={2402.16829},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
    """,
    """
    @article{minderer2022simple,
        title={Simple Open-Vocabulary Object Detection with Vision Transformers},
        author={Matthias Minderer, and Alexey Gritsenko, and Austin Stone, and Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, Neil Houlsby},
        journal={arXiv preprint arXiv:2205.06230},
        year={2022}
    }
    """
]
    factory = BibTeXFactory()
    # Parse a single BibTeX entry
    parsed_entry, flag = factory.parse_bibtex(bibtex_entries[0])
    print("Parsed Single Entry:", parsed_entry)
    print("flag:", flag)
    # Parse multiple BibTeX entries
    parsed_entries = factory.parse_multiple_bibtex_entries(bibtex_entries)
    print("\nParsed Multiple Entries:")
    for entry in parsed_entries:
        print(entry)
