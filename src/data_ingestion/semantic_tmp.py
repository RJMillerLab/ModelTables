import requests, time, json

class ImprovedSemanticScholarAPI:
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
    
    def fetch_paper_by_identifier(self, identifier: str):
        """
        Attempt direct ID-based lookup for the given identifier.
        - identifier might be "DOI:10.48550/arxiv.2212.04356"
          or "arXiv:2212.04356", etc.
        """
        url = f"{self.BASE_URL}/{identifier}?fields=title,authors,year,venue,isOpenAccess,openAccessPdf,externalIds"
        print("[DEBUG] Exact lookup:", url)
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None
    
    def search_fuzzy(self, query: str):
        """
        Fallback fuzzy search if direct lookup fails.
        e.g. "query=10.48550/arxiv.2212.04356"
        """
        search_url = f"{self.BASE_URL}/search"
        params = {
            "query": query,
            "fields": "title,authors,year,venue,isOpenAccess,openAccessPdf,externalIds"
        }
        response = requests.get(search_url, params=params)
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data:
                return data[0]
        return None

    def get_references(self, paper_id: str):
        """
        Once you have the correct 'paper_id', fetch references.
        """
        url = f"{self.BASE_URL}/{paper_id}/references"
        resp = requests.get(url)
        return resp.json() if resp.status_code == 200 else None

    def get_citations(self, paper_id: str):
        """
        Once you have the correct 'paper_id', fetch citations.
        """
        url = f"{self.BASE_URL}/{paper_id}/citations"
        resp = requests.get(url)
        return resp.json() if resp.status_code == 200 else None

    def get_paper(self, raw_doi_or_arxiv=None, fallback_title=None):
        """
        Overall logic:
          1) Try direct ID-based calls with "DOI:<...>" or "arXiv:<...>"
          2) If that fails, do fuzzy search
          3) Then get references & citations with the final paper ID
        """
        # 1. Convert "10.48550/arxiv.2212.04356" to "DOI:10.48550/arxiv.2212.04356" if it looks like a DOI
        #    Or to "arXiv:2212.04356" if it looks like an arXiv ID
        #    This is just heuristic; you can do more robust parsing:
        exact_id = None
        if raw_doi_or_arxiv:
            if raw_doi_or_arxiv.startswith("10."):
                exact_id = f"DOI:{raw_doi_or_arxiv}"
            elif raw_doi_or_arxiv.lower().startswith("arxiv:"):
                exact_id = raw_doi_or_arxiv
            else:
                # Possibly it's already "arXiv:xxxx"
                pass

        paper_json = None
        if exact_id:
            paper_json = self.fetch_paper_by_identifier(exact_id)
        
        # If direct ID-based lookup fails, fallback to fuzzy search
        if not paper_json and raw_doi_or_arxiv:
            paper_json = self.search_fuzzy(raw_doi_or_arxiv)
        
        # If still no success, fallback to title
        if not paper_json and fallback_title:
            paper_json = self.search_fuzzy(fallback_title)

        # if still not found, return None
        if not paper_json:
            print("No results found after exact + fuzzy attempts.")
            return None
        
        # Now that we have the correct JSON, let's get references, citations
        final_id = paper_json.get("paperId") or paper_json.get("externalIds", {}).get("DOI")
        if not final_id:
            print("Paper found but no recognized 'paperId' to fetch references/citations.")
            return None
        
        references_json = self.get_references(final_id)
        citations_json = self.get_citations(final_id)

        return {
            "info": paper_json,
            "references": references_json,
            "citations": citations_json
        }


def example_usage():
    # Example:
    api = ImprovedSemanticScholarAPI()

    # If you suspect "10.48550/arxiv.2212.04356" is a valid arXiv-based DOI
    # and the actual arXiv ID is "arXiv:2212.04356", try the logic:
    results = api.get_paper(
        raw_doi_or_arxiv="10.48550/arxiv.2212.04356",
        fallback_title="Attention Is All You Need"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    example_usage()

