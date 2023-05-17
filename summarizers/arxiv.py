import json
import logging
import os
import re

import arxiv
from arxiv import SortCriterion, SortOrder
import textdistance

from .utils import get_cache_dir, add_progress_task, get_progress_bar, pdf_to_text

logger = logging.getLogger(__name__)


def match_arxiv_id(path):
    pattern = "(?:(?:https?://)?(?:arxiv.org)(?:\/\w+\/))?(\d{4}\.\d{4,5}(v\d*)?)"
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
        arxiv_id = match_arxiv_id(metadata["entry_id"])
        metadata_path = arxiv_metadata_path(arxiv_id, arxiv_path, create_dir=True)
        with open(metadata_path, "w") as fh:
            json.dump(metadata, fh)

    return metadata, paper


def remove_article_abstract(text, abstract):
    if isinstance(text, list):
        text = "".join(text)
    paragraphs = text.split("\n\n")
    abstract_idx = None
    abstract = abstract.split(" ")
    for idx, par in enumerate(paragraphs):
        dist = textdistance.lcsseq.distance(par.split(" "), abstract)
        if dist < 0:
            abstract_idx = idx
        elif abstract_idx:
            break
    if abstract_idx:
        text = paragraphs[abstract_idx + 1 :]
        text = "\n\n".join(text)
    else:
        text = None
    return text


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
    max_results=None,
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
        search = arxiv.Search(
            query=query,
            id_list=id_list,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        results = [r for r in search.results()]
        progress = get_progress_bar()
        task = add_progress_task(
            progress,
            f"Loading articles from arXiv...",
            total=len(results),
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
