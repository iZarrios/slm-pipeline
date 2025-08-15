"""
This is what should be done for more "serious/complete" web scraping.
This is a summary of what should be done to almost never get blocked/banned by any website.

- Infrastructure is key: use distributed IPs (EC2 instances, proxies, or similar) to avoid bans.
- Separate discovery infra from scraping infra to reduce detection risk.
- Avoid aggressive scraping: randomize requests, use delays, respect target's resources.
- Follow website TOS and scrape ethically — excessive scraping can resemble a DDoS.
- For HTTP headers: copy real browser requests via "Inspect > Network tab > Copy as cURL" for authenticity.
- Keep organized: track scraped URLs and statuses to handle failures.
- Basic scraping flow:
    1. Gather URLs
    2. Store them
    3. Iterate through list
    4. Fetch content
    5. Store results
- Playwright/Selenium recommended for dynamic sites (JS-heavy, complex DOMs).

Suggestions for improvement:
- Implement robust retry logic (exponential backoff).
- Use rotating residential or datacenter proxies for wider IP diversity.
- Add request randomization (user agents, timings) to mimic human behavior.
- Automate logging of successes/failures to a database.
- Add checkpointing so interrupted runs can resume without re-scraping.
- Respect robots.txt and set concurrency limits to avoid overloading targets.
"""

import random
import time
from typing import TypedDict, Union

import requests
from bs4 import BeautifulSoup


class SuccessResult(TypedDict):
    text: str
    source: str


class ErrorResult(TypedDict):
    error: str
    source: str


def scrape_with_attribution(url: str) -> Union[SuccessResult, ErrorResult]:
    """
    Scrapes a webpage, handles errors, and adds attribution.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        Union[SuccessResult, ErrorResult]: A dictionary containing the scraped text and source URL
            on success, or an error message and source URL on failure.
    """

    # many websites block requests from scripts that don't look like a real browser. Adding a User-Agent header to requests call can help avoid being blocked.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # introduce a small, random delay to be respectful of the server
        time.sleep(random.uniform(1, 3))
        response = requests.get(url, headers=headers, timeout=10)
        # raise an HTTPError for bad responses (4xx or 5xx)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return {"text": text, "source": url}
    except requests.exceptions.RequestException as e:
        return {"error": f"An error occurred: {e}", "source": url}
