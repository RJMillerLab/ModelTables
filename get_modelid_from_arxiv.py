#!/usr/bin/env python
"""
Get Hugging Face Model ID from arXiv link (supports fuzzy matching for multiple formats)

Usage:
    python get_modelid_from_arxiv.py "https://arxiv.org/abs/1910.09700"
    python get_modelid_from_arxiv.py "1910.09700"
    python get_modelid_from_arxiv.py "https://arxiv.org/pdf/1910.09700.pdf"
    python get_modelid_from_arxiv.py "https://arxiv.org/html/1910.09700v2"
    python get_modelid_from_arxiv.py "https://doi.org/10.48550/arXiv.1910.09700"

Supported input formats:
- https://arxiv.org/abs/1910.09700
- https://arxiv.org/abs/1910.09700v2
- https://arxiv.org/pdf/1910.09700.pdf
- https://arxiv.org/html/1910.09700
- arxiv.org/abs/1910.09700
- arxiv:1910.09700
- 1910.09700
- 1910.09700v2
- DOI: 10.48550/arXiv.1910.09700
- https://doi.org/10.48550/arXiv.1910.09700

This script will:
1. Extract arXiv ID from input URL/DOI/ID (supports multiple formats)
2. Generate multiple possible URL format variants (abs/pdf/html/DOI etc.)
3. Use fuzzy matching in parquet files to find model cards containing the arXiv link
4. Return the corresponding modelId list
"""
import argparse
import re
import sys
from typing import List, Optional

from src.data_analysis.get_from import generic_get_attr_from_attr


def extract_arxiv_id(url_or_id: str) -> Optional[str]:
    """
    Extract arXiv ID from arXiv URL, DOI, or direct ID input (flexible matching)
    
    Supported formats:
    - https://arxiv.org/abs/1910.09700
    - https://arxiv.org/abs/1910.09700v2
    - https://arxiv.org/pdf/1910.09700.pdf
    - https://arxiv.org/pdf/1910.09700v2.pdf
    - https://arxiv.org/html/1910.09700
    - https://arxiv.org/html/1910.09700v2
    - arxiv.org/abs/1910.09700
    - arxiv:1910.09700
    - 1910.09700
    - 1910.09700v2
    - DOI: 10.48550/arXiv.1910.09700
    - https://doi.org/10.48550/arXiv.1910.09700
    """
    if not url_or_id:
        return None
    
    # Remove leading/trailing whitespace
    url_or_id = url_or_id.strip()
    
    # If it's a pure numeric format (YYYY.NNNNN or YYYY.NNNN), return directly
    if re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', url_or_id):
        return url_or_id
    
    # Handle DOI format (may contain arXiv ID)
    # Format: 10.48550/arXiv.1910.09700 or doi.org/10.48550/arXiv.1910.09700
    doi_patterns = [
        r'10\.\d+/arXiv\.(\d{4}\.\d{4,5}(?:v\d+)?)',
        r'doi\.org/10\.\d+/arXiv\.(\d{4}\.\d{4,5}(?:v\d+)?)',
        r'arXiv\.(\d{4}\.\d{4,5}(?:v\d+)?)',
    ]
    for pattern in doi_patterns:
        match = re.search(pattern, url_or_id, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Extract from various arXiv URL formats
    # Supports: abs, pdf, html, and other possible paths
    arxiv_patterns = [
        # Standard format: arxiv.org/abs/xxx, arxiv.org/pdf/xxx, arxiv.org/html/xxx
        r'arxiv\.org/(?:abs|pdf|html)/([\d\.]+(?:v\d+)?)',
        # arxiv: prefix
        r'arxiv[:/](\d{4}\.\d{4,5}(?:v\d+)?)',
        # Direct numeric format
        r'(\d{4}\.\d{4,5}(?:v\d+)?)',
    ]
    
    for pattern in arxiv_patterns:
        match = re.search(pattern, url_or_id, re.IGNORECASE)
        if match:
            arxiv_id = match.group(1)
            # If it's a PDF link, remove possible .pdf suffix
            if arxiv_id.endswith('.pdf'):
                arxiv_id = arxiv_id[:-4]
            # Validate format correctness
            if re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', arxiv_id):
                return arxiv_id
    
    return None


def normalize_arxiv_url(arxiv_id: str) -> List[str]:
    """
    Generate multiple possible arXiv URL formats for fuzzy matching
    
    Includes:
    - Variants with and without version numbers
    - Different paths: abs/pdf/html
    - With/without https:// protocol
    - arxiv: prefix format
    - DOI format
    """
    # Remove version number (if exists) for matching
    base_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
    
    urls = []
    
    # 1. Standard URL formats (with version number)
    urls.extend([
        f"https://arxiv.org/abs/{arxiv_id}",
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        f"https://arxiv.org/html/{arxiv_id}",
        f"arxiv.org/abs/{arxiv_id}",
        f"arxiv.org/pdf/{arxiv_id}.pdf",
        f"arxiv.org/html/{arxiv_id}",
    ])
    
    # 2. URL formats without version number
    urls.extend([
        f"https://arxiv.org/abs/{base_id}",
        f"https://arxiv.org/pdf/{base_id}.pdf",
        f"https://arxiv.org/html/{base_id}",
        f"arxiv.org/abs/{base_id}",
        f"arxiv.org/pdf/{base_id}.pdf",
        f"arxiv.org/html/{base_id}",
    ])
    
    # 3. arxiv: prefix format
    urls.extend([
        f"arxiv:{arxiv_id}",
        f"arxiv:{base_id}",
    ])
    
    # 4. Direct ID format
    urls.extend([
        arxiv_id,
        base_id,
    ])
    
    # 5. DOI format (if possible)
    # Note: Not all arXiv papers have DOI, but we try to match
    urls.extend([
        f"10.48550/arXiv.{arxiv_id}",
        f"10.48550/arXiv.{base_id}",
        f"https://doi.org/10.48550/arXiv.{arxiv_id}",
        f"https://doi.org/10.48550/arXiv.{base_id}",
        f"doi.org/10.48550/arXiv.{arxiv_id}",
        f"doi.org/10.48550/arXiv.{base_id}",
    ])
    
    # 6. Partial matching variants (for fuzzy matching)
    # Use only part of the ID to allow get_from's LIKE matching to find URLs containing the ID
    if len(base_id) >= 8:  # At least YYYY.NNNN format
        # Extract year part for partial matching
        year_part = base_id[:4] if len(base_id) >= 4 else ""
        if year_part:
            urls.append(year_part)  # Only match year part
    
    return urls


def get_modelids_from_arxiv(arxiv_input: str, debug: bool = False) -> List[str]:
    """
    Get model IDs from arXiv link (supports fuzzy matching for multiple formats)
    
    Args:
        arxiv_input: arXiv URL, DOI, or ID
        debug: Whether to print debug information
    
    Returns:
        List of found modelId
    """
    # 1. Extract arXiv ID
    arxiv_id = extract_arxiv_id(arxiv_input)
    if not arxiv_id:
        print(f"Warning: Could not extract standard arXiv ID from input '{arxiv_input}'")
        print("Will try to use the original input directly for fuzzy matching...")
        # Even if extraction fails, try to use original input directly (may contain partial match)
        arxiv_id = arxiv_input.strip()
        if debug:
            print(f"Using original input for matching: {arxiv_id}")
    else:
        if debug:
            print(f"Extracted arXiv ID: {arxiv_id}")
    
    # 2. Generate multiple possible URL formats
    possible_urls = normalize_arxiv_url(arxiv_id)
    if debug:
        print(f"Generated {len(possible_urls)} possible matching formats")
        print(f"Sample formats (first 10): {possible_urls[:10]}")
    
    # 3. Query for each possible URL format
    # get_from uses LIKE matching, so it can find even if not exactly matching
    all_results = []
    seen_results = set()  # Avoid duplicate additions
    
    for url in possible_urls:
        if debug:
            print(f"\nQuerying: {url[:80]}..." if len(url) > 80 else f"\nQuerying: {url}")
        results = generic_get_attr_from_attr(
            target_attr="modelId",
            source_attr="pdf_link",
            value=url,
            debug=debug
        )
        if results:
            new_results = [r for r in results if r not in seen_results]
            all_results.extend(new_results)
            seen_results.update(new_results)
            if debug:
                print(f"  Found {len(new_results)} new results (total: {len(all_results)})")
    
    # 4. Deduplicate (double check)
    unique_results = list(set(all_results))
    
    if debug:
        print(f"\nTotal found {len(unique_results)} unique model ID(s)")
    
    return unique_results


def main():
    parser = argparse.ArgumentParser(
        description="Get Hugging Face Model ID from arXiv link",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard arXiv URL
  python get_modelid_from_arxiv.py "https://arxiv.org/abs/1910.09700"
  
  # Direct ID input
  python get_modelid_from_arxiv.py "1910.09700"
  
  # PDF link
  python get_modelid_from_arxiv.py "https://arxiv.org/pdf/1910.09700.pdf"
  
  # HTML link (with version number)
  python get_modelid_from_arxiv.py "https://arxiv.org/html/1910.09700v2"
  
  # DOI format
  python get_modelid_from_arxiv.py "https://doi.org/10.48550/arXiv.1910.09700"
  
  # Debug mode
  python get_modelid_from_arxiv.py "1910.09700" --debug
        """
    )
    parser.add_argument(
        "arxiv_link",
        type=str,
        help="arXiv URL, DOI, or ID (supports multiple formats, e.g.: https://arxiv.org/abs/1910.09700, 1910.09700, https://arxiv.org/html/1910.09700v2, DOI:10.48550/arXiv.1910.09700)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information"
    )
    parser.add_argument(
        "--schema-log",
        type=str,
        default="logs/parquet_schema.log",
        help="Path to parquet schema log file (default: logs/parquet_schema.log)"
    )
    
    args = parser.parse_args()
    
    # Get model IDs
    model_ids = get_modelids_from_arxiv(args.arxiv_link, debug=args.debug)
    
    if not model_ids:
        print(f"\nNo model ID found associated with arXiv '{args.arxiv_link}'")
        print("\nTips:")
        print("  - Verify the arXiv ID is correct")
        print("  - This paper may not have a corresponding Hugging Face model")
        print("  - Try using --debug to see detailed query process")
        sys.exit(1)
    
    # Output results
    print(f"\nFound {len(model_ids)} model ID(s):")
    for i, model_id in enumerate(model_ids, 1):
        print(f"  {i}. {model_id}")
    
    # If only one result, output it for easy script usage
    if len(model_ids) == 1:
        print(f"\n{model_ids[0]}")


if __name__ == "__main__":
    main()

