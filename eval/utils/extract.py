import re


def extract(string, tag):
    pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
    match = re.search(pattern, string, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""