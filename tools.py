import requests

from bs4 import BeautifulSoup
from ddgs import DDGS
from crewai.tools import tool


# =========================================================
# General Web Search
# =========================================================

@tool("Web Search")
def web_search(query: str) -> str:
    """
    Search the public web for relevant information.

    Returns titles, URLs and snippets.
    """

    try:

        results = DDGS().text(
            query,
            max_results=8,
        )

        formatted = []

        for item in results:

            title = item.get("title", "No title")
            url = item.get("href", "")
            body = item.get("body", "")

            formatted.append(
                f"TITLE: {title}\n"
                f"URL: {url}\n"
                f"SUMMARY: {body}\n"
            )

        if not formatted:
            return "No web search results were found."

        return "\n---\n".join(formatted)

    except Exception as exc:

        return f"Web search failed: {exc}"


# =========================================================
# Academic Search
# =========================================================

@tool("Academic Search")
def academic_search(query: str) -> str:
    """
    Search Crossref for academic publications.

    Crossref provides publication metadata such as:
    - Title
    - Authors
    - Publication date
    - DOI
    - Publisher
    """

    try:

        response = requests.get(
            "https://api.crossref.org/works",
            params={
                "query.bibliographic": query,
                "rows": 8,
                "select": (
                    "title,author,published,DOI,"
                    "URL,container-title,publisher"
                ),
            },
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        items = data.get("message", {}).get("items", [])

        if not items:
            return "No academic publications were found."

        formatted = []

        for item in items:

            titles = item.get("title", [])

            title = (
                titles[0]
                if titles
                else "Unknown title"
            )

            authors = []

            for author in item.get("author", []):

                given = author.get("given", "")
                family = author.get("family", "")

                full_name = f"{given} {family}".strip()

                if full_name:
                    authors.append(full_name)

            author_text = (
                ", ".join(authors)
                if authors
                else "Authors not listed"
            )

            doi = item.get("DOI", "No DOI")

            url = item.get(
                "URL",
                "",
            )

            publisher = item.get(
                "publisher",
                "Publisher not listed",
            )

            journal = item.get(
                "container-title",
                [],
            )

            journal_name = (
                journal[0]
                if journal
                else "Journal not listed"
            )

            published = item.get(
                "published",
                {},
            )

            date_parts = published.get(
                "date-parts",
                [],
            )

            year = ""

            if date_parts and date_parts[0]:
                year = str(date_parts[0][0])

            formatted.append(
                f"TITLE: {title}\n"
                f"AUTHORS: {author_text}\n"
                f"YEAR: {year or 'Unknown'}\n"
                f"JOURNAL: {journal_name}\n"
                f"PUBLISHER: {publisher}\n"
                f"DOI: {doi}\n"
                f"URL: {url}\n"
            )

        return "\n---\n".join(formatted)

    except Exception as exc:

        return f"Academic search failed: {exc}"


# =========================================================
# Industry Search
# =========================================================

@tool("Industry Search")
def industry_search(query: str) -> str:
    """
    Search for commercial, industry and market information.
    """

    industry_query = (
        f"{query} "
        "companies startups products market industry commercial solution"
    )

    try:

        results = DDGS().text(
            industry_query,
            max_results=10,
        )

        formatted = []

        for item in results:

            title = item.get(
                "title",
                "No title",
            )

            url = item.get(
                "href",
                "",
            )

            body = item.get(
                "body",
                "",
            )

            formatted.append(
                f"TITLE: {title}\n"
                f"URL: {url}\n"
                f"SUMMARY: {body}\n"
            )

        if not formatted:
            return "No industry search results were found."

        return "\n---\n".join(formatted)

    except Exception as exc:

        return f"Industry search failed: {exc}"


# =========================================================
# Web Page Reader
# =========================================================

@tool("Read Web Page")
def read_web_page(url: str) -> str:
    """
    Download a public web page and extract readable text.
    """

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Research Multi-Agent)"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=20,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        # Remove non-content elements
        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "nav",
                "footer",
            ]
        ):
            element.decompose()

        text = soup.get_text(
            separator=" ",
        )

        text = " ".join(
            text.split()
        )

        # Prevent extremely large pages
        return text[:12000]

    except Exception as exc:

        return f"Unable to read webpage: {exc}"
