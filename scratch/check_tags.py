import re

def check_html_blocks():
    with open('streamlit_app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all multi-line strings
    matches_triple = re.findall(r'\"\"\"([\s\S]*?)\"\"\"', content)
    for i, match in enumerate(matches_triple):
        if '<div' in match or '</div' in match:
            opens = len(re.findall(r'<div', match))
            closes = len(re.findall(r'</div', match))
            if opens != closes:
                print(f"Mismatched Triple-Quote Block {i}: opens={opens}, closes={closes}")
                print(match)
                print("="*80)

    # Find all single line double-quoted strings
    matches_double = re.findall(r'\"([^\n\"]*?)\"', content)
    for i, match in enumerate(matches_double):
        if '<div' in match or '</div' in match:
            opens = len(re.findall(r'<div', match))
            closes = len(re.findall(r'</div', match))
            if opens != closes:
                print(f"Mismatched Double-Quote Block {i}: opens={opens}, closes={closes}")
                print(match)
                print("="*80)

    # Find all single line single-quoted strings
    matches_single = re.findall(r'\'([^\n\']*?)\'', content)
    for i, match in enumerate(matches_single):
        if '<div' in match or '</div' in match:
            opens = len(re.findall(r'<div', match))
            closes = len(re.findall(r'</div', match))
            if opens != closes:
                print(f"Mismatched Single-Quote Block {i}: opens={opens}, closes={closes}")
                print(match)
                print("="*80)

if __name__ == '__main__':
    check_html_blocks()
