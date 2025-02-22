"""
Author: Zhengyuan Dong
Date: 2025-02-15
Description:
    This script provides a simple interface to:
    - Fuzzily fetch paper information (via DOI first, else Title)
    - Retrieve references (papers it cites) and citations (papers citing it)
    - Demonstrate how to batch-process records from a DataFrame

Usage:
    python citation_fetcher.py
"""

from abc import ABC, abstractmethod
import requests
import json
import time
import pandas as pd


class AcademicAPI(ABC):
    """Abstract base class for academic APIs."""

    @abstractmethod
    def search_paper(self, query: str):
        """Search for a paper using fuzzy search (DOI or title)."""
        pass

    @abstractmethod
    def get_info(self, identifier: str):
        """Retrieve paper information given an identifier (DOI, Paper ID)."""
        pass

    @abstractmethod
    def get_references(self, identifier: str):
        """Retrieve references of a given paper."""
        pass

    @abstractmethod
    def get_citations(self, identifier: str):
        """Retrieve citations of a given paper."""
        pass


class SemanticScholarAPI(AcademicAPI):
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
    SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search_paper(self, query: str):
        """Fuzzy search by DOI or title."""
        params = {
            "query": query,
            "fields": "title,authors,year,venue,isOpenAccess,openAccessPdf,externalIds"
        }
        response = requests.get(self.SEARCH_URL, params=params)
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data:
                return data[0]  # Return the first (best) match
        return None

    def get_info(self, identifier: str):
        """Given a Semantic Scholar paper ID or DOI, fetch paper info."""
        url = f"{self.BASE_URL}/{identifier}?fields=title,authors,year,venue,isOpenAccess,openAccessPdf"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None

    def get_references(self, identifier: str):
        url = f"{self.BASE_URL}/{identifier}/references"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None

    def get_citations(self, identifier: str):
        url = f"{self.BASE_URL}/{identifier}/citations"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None


class OpenAlexAPI(AcademicAPI):
    BASE_URL = "https://api.openalex.org/works"

    def search_paper(self, query: str):
        """Fuzzy search by DOI or title."""
        url = f"{self.BASE_URL}?search={query}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get("results", [])
            if data:
                return data[0]  # Return the first (best) match
        return None

    def get_info(self, identifier: str):
        """Given an OpenAlex ID, fetch paper info."""
        url = f"{self.BASE_URL}/{identifier}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                "title": data.get("display_name"),
                "year": data.get("publication_year"),
                "doi": data.get("ids", {}).get("doi"),
                "openalex_id": data.get("id"),
                "cited_by_count": data.get("cited_by_count")
            }
        return None

    def get_references(self, identifier: str):
        """Given an OpenAlex ID, fetch references."""
        url = f"{self.BASE_URL}/{identifier}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            reference_ids = data.get("referenced_works", [])
            return self.get_paper_details(reference_ids)
        return None

    def get_citations(self, identifier: str):
        """Given an OpenAlex ID, fetch citations."""
        url = f"{self.BASE_URL}?filter=cites:{identifier}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            citation_ids = [paper["id"] for paper in data.get("results", [])]
            return self.get_paper_details(citation_ids)
        return None

    def get_paper_details(self, paper_ids: list):
        """Helper to fetch multiple paper details."""
        papers = []
        for paper_id in paper_ids:
            url = f"{self.BASE_URL}/{paper_id}"
            response = requests.get(url)
            if response.status_code == 200:
                paper_data = response.json()
                papers.append({
                    "paperId": paper_data.get("id"),
                    "title": paper_data.get("display_name")
                })
            time.sleep(0.2)  # Sleep to avoid rate-limits
        return papers


class AcademicAPIFactory:
    """Factory to get the correct API class by name."""

    @staticmethod
    def get_api(api_name: str):
        if api_name == 'semantic_scholar':
            return SemanticScholarAPI()
        elif api_name == 'openalex':
            return OpenAlexAPI()
        else:
            raise ValueError(f"Unknown API: {api_name}")

def search_and_fetch_info(doi=None, title=None, api_priority=["semantic_scholar", "openalex"]):
    """
    1) Searches for a paper by DOI first (if provided),
       then by Title (if DOI search fails or not found).
    2) Prioritizes the list of APIs in api_priority order.
    3) Returns a dict with 'info', 'references', 'citations'.
    """
    best_paper = None
    best_api = None
    paper_id = None
    
    if doi and doi.startswith("10."):
        doi = f"DOI:{doi}"

    # Try all APIs in the given priority (e.g., semantic_scholar -> openalex)
    for api_name in api_priority:
        api = AcademicAPIFactory.get_api(api_name)

        # Try searching by DOI first
        if doi:
            # print(f"[DEBUG] Searching by DOI: {doi} using {api_name}")
            paper = api.search_paper(doi)
            if paper:
                best_paper = paper
                best_api = api
                # Different APIs store their unique IDs differently
                # For Semantic Scholar, it's "externalIds" -> "DOI" or "paperId"
                # For OpenAlex, it's "id"
                if api_name == "semantic_scholar":
                    paper_id = paper.get("externalIds", {}).get("DOI") or paper.get("paperId")
                else:
                    paper_id = paper.get("id")
                break  # Found a match by DOI; stop searching

        # If no match by DOI, try searching by title (if provided)
        if not best_paper and title:
            # print(f"[DEBUG] Searching by Title: {title} using {api_name}")
            paper = api.search_paper(title)
            if paper:
                best_paper = paper
                best_api = api
                if api_name == "semantic_scholar":
                    paper_id = paper.get("externalIds", {}).get("DOI") or paper.get("paperId")
                else:
                    paper_id = paper.get("id")

        # If we've found a best_paper, we don't continue to the next API
        if best_paper:
            break

    if not best_paper:
        print("No results found.\n")
        return None

    # We have a match; now fetch info, references, citations
    if not paper_id:
        print("Warning: Found paper, but no valid ID to retrieve details.\n")
        return None

    info = best_api.get_info(paper_id)
    references = best_api.get_references(paper_id)
    citations = best_api.get_citations(paper_id)

    return {
        "paper_id": paper_id,
        "info": info,
        "references": references,
        "citations": citations
    }


# ------------------------------
# BATCH PROCESSING LOGIC
# ------------------------------

def batch_fetch_info(df, doi_col="doi", title_col="title", max_records=5):
    """
    Example function to batch-fetch from a DataFrame (df) where each row
    contains a possible 'doi' or 'title' (or both).
    
    :param df: Pandas DataFrame with columns [doi_col, title_col]
    :param doi_col: Name of the column containing a DOI
    :param title_col: Name of the column containing a Title
    :param max_records: Limit how many rows to process for demonstration
    
    :return: A list of results (dicts) for each row. 
             Each dict has 'row_index', 'doi', 'title', 'info', 'references', 'citations'.
    """
    results = []
    # For demonstration, let's limit the number of rows
    subset = df.head(max_records)

    for idx, row in subset.iterrows():
        # Extract relevant fields
        row_doi = row.get(doi_col, None)
        row_title = row.get(title_col, None)

        print(f"\n[Row {idx}] Searching for: DOI='{row_doi}', TITLE='{row_title}'...")
        fetch_result = search_and_fetch_info(
            doi=row_doi if pd.notnull(row_doi) else None,
            title=row_title if pd.notnull(row_title) else None
        )

        # Store the results in a structured way
        out = {
            "row_index": idx,
            "doi": row_doi,
            "title": row_title,
            "info": None,
            "references": None,
            "citations": None
        }
        if fetch_result:
            out["info"] = fetch_result["info"]
            out["references"] = fetch_result["references"]
            out["citations"] = fetch_result["citations"]

        results.append(out)

    return results


# ------------------------------
# TEST EXAMPLES
# ------------------------------

if __name__ == "__main__":
    # 1. Single search with both DOI and Title
    # openalex is better than semantic scholar here
    print("\nTEST CASE 1: Searching for '10.48550/arxiv.2212.04356' + 'Robust Speech Recognition via Large-Scale Weak Supervision'...\n")
    result1 = search_and_fetch_info(doi="10.48550/arxiv.2212.04356", title="Robust Speech Recognition via Large-Scale Weak Supervision")
    print("RESULT 1:")
    print(json.dumps(result1, indent=4))
    # 2. Single search with Title only
    # semantic scholar is better than openalex here
    print("\nTEST CASE 2: Searching with Title Only -> 'Robust Speech Recognition via Large-Scale Weak Supervision'...\n")
    result2 = search_and_fetch_info(title="Robust Speech Recognition via Large-Scale Weak Supervision")
    print("RESULT 2:")
    print(json.dumps(result2, indent=4))
    # 3. Single search with DOI only
    print("\nTEST CASE 3: Searching with DOI Only -> '10.48550/arxiv.2212.04356'...\n")
    result3 = search_and_fetch_info(doi="10.48550/arxiv.2212.04356")
    print("RESULT 3:")
    print(json.dumps(result3, indent=4))
    # 4. Batch testing on a small DataFrame
    print("\nTEST CASE 4: BATCH PROCESSING on a small DataFrame...\n")
    data = [
        {"doi": "10.48550/arxiv.2212.04356", "title": "Robust Speech Recognition via Large-Scale Weak Supervision"},
        {"doi": None, "title": "ChatGPT: Past, Present, Future"},
        {"doi": "10.18653/v1/N19-1423", "title": None},  # Another example
    ]
    df_test = pd.DataFrame(data)
    batch_results = batch_fetch_info(df_test, doi_col="doi", title_col="title", max_records=3)
    print("\nBATCH RESULTS:")
    print(json.dumps(batch_results, indent=4))
    