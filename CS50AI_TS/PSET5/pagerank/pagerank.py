import os
import random
import re
import sys

# Constants
DAMPING = 0.85
SAMPLES = 10000


def main():
    """
    Main function to compute and display PageRank results using
    both sampling and iteration methods.
    """
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")

    corpus = crawl(sys.argv[1])

    print("\nPageRank Results from Sampling (n = {SAMPLES})")
    sampling_ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    display_ranks(sampling_ranks)

    print("\nPageRank Results from Iteration")
    iterative_ranks = iterate_pagerank(corpus, DAMPING)
    display_ranks(iterative_ranks)


def crawl(directory):
    """
    Parse a directory of HTML pages to find links.
    Returns a dictionary mapping pages to their outgoing links.
    """
    pages = {}
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            with open(os.path.join(directory, filename)) as f:
                links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", f.read())
                pages[filename] = set(links) - {filename}

    # Filter links to include only valid pages
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)
    return pages


def transition_model(corpus, page, damping_factor):
    """
    Generate the transition model for a given page based on damping factor.
    """
    num_pages = len(corpus)
    probabilities = {key: (1 - damping_factor) / num_pages for key in corpus}

    links = corpus[page]
    if links:
        for link in links:
            probabilities[link] += damping_factor / len(links)
    else:
        # Uniform distribution for pages without outgoing links
        probabilities = {key: 1 / num_pages for key in corpus}

    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Calculate PageRank using sampling method.
    """
    ranks = {page: 0 for page in corpus}
    page = random.choice(list(corpus.keys()))

    for _ in range(n):
        ranks[page] += 1
        transition = transition_model(corpus, page, damping_factor)
        page = random.choices(list(transition.keys()), weights=transition.values(), k=1)[0]

    # Normalize results
    total = sum(ranks.values())
    return {page: rank / total for page, rank in ranks.items()}


def iterate_pagerank(corpus, damping_factor, tolerance=0.001):
    """
    Calculate PageRank using iteration method until convergence.
    """
    num_pages = len(corpus)
    ranks = {page: 1 / num_pages for page in corpus}

    while True:
        new_ranks = {}
        for page in corpus:
            rank = (1 - damping_factor) / num_pages
            rank += damping_factor * sum(
                ranks[link] / (len(corpus[link]) or num_pages)
                for link in corpus if page in corpus[link] or not corpus[link]
            )
            new_ranks[page] = rank

        # Check for convergence
        if all(abs(new_ranks[page] - ranks[page]) < tolerance for page in ranks):
            break
        ranks = new_ranks

    return ranks


def display_ranks(ranks):
    """
    Helper function to display PageRank results.
    """
    for page, rank in sorted(ranks.items()):
        print(f"  {page}: {rank:.4f}")


if __name__ == "__main__":
    main()
