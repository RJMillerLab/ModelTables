import requests
import feedparser
import os

def fetch_arxiv_entries(query, max_results=5):
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'search_query={query}&start=0&max_results={max_results}'
    response = requests.get(base_url + search_query)
    response.raise_for_status()
    return feedparser.parse(response.content)

def download_tex_source(arxiv_id, download_dir='downloads'):
    src_url = f'https://arxiv.org/e-print/{arxiv_id}'
    response = requests.get(src_url, stream=True)
    response.raise_for_status()
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    arxiv_id = arxiv_id.replace('/', '_')
    file_path = os.path.join(download_dir, f'{arxiv_id}.tar.gz')
    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    #print(f'Downloaded TeX source for {arxiv_id} to {file_path}')

def main():
    query = 'all:quantum computing'  # Replace with your search query
    entries = fetch_arxiv_entries(query)
    for entry in entries.entries:
        arxiv_id = entry.id.split('/abs/')[-1]
        print(f'Title: {entry.title}')
        print(f'Authors: {", ".join(author.name for author in entry.authors)}')
        print(f'Published: {entry.published}')
        print(f'Primary Category: {entry.arxiv_primary_category["term"]}')
        print(f'Link to PDF: {entry.link.replace("/abs/", "/pdf/")}')
        print('Downloading TeX source...')
        print('arxiv_id: ', arxiv_id)
        download_tex_source(arxiv_id)
        print('---')

if __name__ == '__main__':
    main()

