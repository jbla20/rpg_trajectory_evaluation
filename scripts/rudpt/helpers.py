# Standard library imports
import re
from contextlib import contextmanager
import sys, os
import time

# Type imports
from typing import List, Union

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

def latexify(text : Union[str, List[str]]) -> Union[str, List[str]]:
    # Dictionary of LaTeX special characters and their escaped versions
    special_characters = {
        '_': r'\_',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '&': r'\&',
        '{': r'\{',
        '}': r'\}',
        '^': r'\^',
        '~': r'\textasciitilde',
        # u'\N{DEGREE SIGN}': r'^{\circ}'
    }

    # Define a function to escape a string
    def escape_string(s: str) -> str:
        # Only escape characters that are not already escaped
        for char, escaped_char in special_characters.items():
            s = re.sub(f'(?<!\\\\){re.escape(char)}', escaped_char, s)
        return s

    # Escape each special character in the string
    if isinstance(text, str):
        return escape_string(text)
    elif isinstance(text, list):
        return [escape_string(t) for t in text]
    else:
        raise ValueError("Invalid input type for latexify: " + type(text))

def unlatexify(text : Union[str, List[str]]) -> Union[str, List[str]]:
    # Dictionary of LaTeX special characters and their unescaped versions
    special_characters = {
        r'\_': '_',
        r'\%': '%',
        r'\$': '$',
        r'\#': '#',
        r'\&': '&',
        r'\{': '{',
        r'\}': '}',
        r'\^': '^',
        r'\textasciitilde': '~',
        # r'^{\circ}': u'\N{DEGREE SIGN}'
    }

    # Remove potential bold formatting
    text = re.sub(r'\\textbf{(.+?)}', r'\1', text)

    # Define function to unescape a string
    def unescape_string(s: str) -> str:
        # Only escape characters that are not already escaped
        for char, escaped_char in special_characters.items():
            s = re.sub(f'(?<!\\\\){re.escape(char)}', escaped_char, s)
        return s

    # Unescape each special character in the string
    if isinstance(text, str):
        return unescape_string(text)
    elif isinstance(text, list):
        return [unescape_string(t) for t in text]
    else:
        raise ValueError("Invalid input type for unlatexify: " + type(text))

def boldify(text : Union[str, List[str]]) -> Union[str, List[str]]:
    text = latexify(text)
    def apply_bold(s: str) -> str:
        # Check if the string is already boldified
        if s.startswith(r'\textbf{') and s.endswith('}'):
            return s
        else:
            return r'\textbf{' + s + '}'

    if isinstance(text, str):
        return apply_bold(text)
    elif isinstance(text, list):
        return [apply_bold(t) for t in text]
    else:
        raise ValueError("Invalid input type for boldify: " + type(text))

