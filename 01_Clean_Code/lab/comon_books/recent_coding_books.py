"""Module for find_recent_coding_books function.

Author: Mikel
Date: May 2022
"""


def find_recent_coding_books(recent_books_path, coding_books_path):
    """Finds common book ids that appear in the lists contained in the passed files.

    Args:
        recent_books_path: (str) path for file containing all recent book records
        coding_books_path: (str) path for file containing all coding book records
    Returns:
        recent_coding_books: (set) all commoon book records: recent and coding
    """
    with open(recent_books_path, encoding="utf-8") as recent_books_file:
        recent_books = recent_books_file.read().split('\n')

    with open(coding_books_path, encoding="utf-8") as coding_books_file:
        coding_books = coding_books_file.read().split('\n')

    recent_coding_books = set(coding_books).intersection(set(recent_books))
    return recent_coding_books


if __name__ == "__main__":

    RECENT_CODING_BOOKS = find_recent_coding_books(
        'books_published_last_two_years.txt', 'all_coding_books.txt')
