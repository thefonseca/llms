import json
import logging
import os
from pathlib import Path
import re

import arxiv
from arxiv import SortCriterion, SortOrder
import numpy as np

from .utils import (
    get_cache_dir,
    add_progress_task,
    get_progress_bar,
    pdf_to_text,
    clean_before_section,
    remove_article_abstract,
)

logger = logging.getLogger(__name__)


def match_arxiv_id(path):
    pattern = r"(?:(?:https?://)?(?:arxiv.org)(?:\/\w+\/))?(\d{4}\.\d{4,5}(v\d*)?)"
    match = re.match(pattern, path)
    if match:
        return match.groups()[0]


def arxiv_paper_to_dict(paper):
    metadata = dict(
        entry_id=paper.entry_id,
        updated=paper.updated.strftime("%m/%d/%Y, %H:%M:%S"),
        published=paper.published.strftime("%m/%d/%Y, %H:%M:%S"),
        title=paper.title,
        authors=[str(x) for x in paper.authors],
        summary=paper.summary,
        comment=paper.comment,
        journal_ref=paper.journal_ref,
        doi=paper.doi,
        primary_category=paper.primary_category,
        categories=paper.categories,
        links=[str(x) for x in paper.links],
        pdf_url=paper.pdf_url,
    )
    return metadata


def arxiv_metadata_path(arxiv_id, arxiv_path, create_dir=False):
    metadata_filename = f"{arxiv_id}.json"
    metadata_dir = os.path.join(arxiv_path, "metadata")
    metadata_path = os.path.join(metadata_dir, metadata_filename)
    if create_dir:
        os.makedirs(metadata_dir, exist_ok=True)
    return metadata_path


def load_arxiv_metadata(arxiv_id, arxiv_path, paper=None):
    metadata = None
    if arxiv_id:
        arxiv_id = match_arxiv_id(arxiv_id)
        metadata_path = arxiv_metadata_path(arxiv_id, arxiv_path, create_dir=True)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as fh:
                metadata = json.load(fh)
        else:
            paper = next(arxiv.Search(id_list=[arxiv_id]).results())

    if paper:
        metadata = arxiv_paper_to_dict(paper)
        metadata_path = arxiv_metadata_path(arxiv_id, arxiv_path, create_dir=True)
        with open(metadata_path, "w") as fh:
            json.dump(metadata, fh)

    return metadata, paper


def load_arxiv_article(
    arxiv_id=None, paper=None, arxiv_path=None, remove_abstract=False
):
    if arxiv_path is None:
        arxiv_path = get_cache_dir("arxiv")

    metadata, paper = load_arxiv_metadata(arxiv_id, arxiv_path, paper=paper)
    arxiv_id = match_arxiv_id(metadata["entry_id"])
    if arxiv_id is None:
        raise ValueError(f"Error matching paper id: {metadata['entry_id']}")

    pdf_filename = f"{arxiv_id}.pdf"
    pdf_dir = os.path.join(arxiv_path, "pdfs")
    pdf_path = os.path.join(pdf_dir, pdf_filename)
    os.makedirs(pdf_dir, exist_ok=True)

    if not os.path.exists(pdf_path):
        if paper is None:
            paper = next(arxiv.Search(id_list=[arxiv_id]).results())
        paper.download_pdf(dirpath=pdf_dir, filename=pdf_filename)

    txt_filename = f"{arxiv_id}.txt"
    txt_path = os.path.join(pdf_dir, txt_filename)

    if not os.path.exists(txt_path) or "text" not in metadata:
        metadata["text"] = pdf_to_text(pdf_path)
        metadata_path = arxiv_metadata_path(arxiv_id, arxiv_path)
        with open(metadata_path, "w") as fh:
            json.dump(metadata, fh)

    if metadata["text"] and remove_abstract:
        text = remove_article_abstract(metadata["text"], metadata["summary"])
        if text is None:
            logger.warning(f"Could not find abstract in article {arxiv_id}")
        metadata["text"] = text

    return metadata


def search_arxiv(
    id_list=None,
    query=None,
    max_results=100,
    sort_by=SortCriterion.Relevance,
    sort_order=SortOrder.Descending,
    remove_abstract=False,
):
    if max_results is None:
        max_results = float("inf")
    if id_list is None:
        id_list = []
    if isinstance(id_list, str):
        id_list = [id_list]

    papers = []

    if query is None:
        for arxiv_id in id_list:
            # if using only IDs, try to load from local cache first
            paper = load_arxiv_article(arxiv_id, remove_abstract=remove_abstract)
            papers.append(paper)
    else:
        # Construct the default API client.
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        results = client.results(search)
        # results = [r for r in search.results()]
        progress = get_progress_bar()
        task = add_progress_task(
            progress,
            f"Loading articles from arXiv...",
            total=max_results,
            existing_ok=False,
        )
        with progress:
            for result in results:
                progress.update(task, description=f"Loading {result.entry_id}...")
                paper = load_arxiv_article(
                    paper=result, remove_abstract=remove_abstract
                )
                if paper["text"]:
                    papers.append(paper)
                progress.update(task, advance=1)

    return papers


def load_arxiv_data(arxiv_id, arxiv_query, max_samples, remove_abstract=False):
    if isinstance(arxiv_id, list):
        arxiv_id = [str(x) for x in arxiv_id]
    else:
        arxiv_id = str(arxiv_id)

    logger.info(f"Arxiv IDs: {arxiv_id}")
    logger.info(f"Arxiv query: {arxiv_query}")
    if arxiv_id and Path(arxiv_id).suffix == ".txt":
        arxiv_ids_file = arxiv_id
        arxiv_id = np.loadtxt(arxiv_ids_file)
        logger.info(f"Loaded {len(arxiv_id)} arXiv IDs from {arxiv_ids_file}")

    if max_samples is None:
        max_samples = float("inf")
    else:
        # add 10% more samples as some of them will not be valid
        max_samples = int(max_samples * 1.1)

    papers = search_arxiv(
        arxiv_id,
        arxiv_query,
        max_results=max_samples,
        sort_by=SortCriterion.SubmittedDate,
        remove_abstract=remove_abstract,
    )

    for p in papers:
        if remove_abstract:
            p["text"] = clean_before_section(p["text"])
        elif p["text"]:
            p["text"] = "\n".join(p["text"])

    papers = [p for p in papers if p["text"]]
    return papers
