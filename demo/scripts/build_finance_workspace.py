"""Insert Finance into the preserved upstream UI without rewriting Chat/Anomaly.

Reference: git-push-kar/Polyvalent master, ui/index.html blob
4fad71c9c8d645a5acc633efef9e50a9407aaa4b (retrieved 2026-11-17).
"""
from pathlib import Path
import re

ui = Path(__file__).resolve().parents[1] / 'ui'
page = (ui / 'reference/polyvalent-original-index.html').read_text(encoding='utf-8')
fragment = (ui / 'finance-workspace.html').read_text(encoding='utf-8')
page, count = re.subn(
    r'(<section class="view" id="view-finance"[^>]*>).*?(</section>)',
    lambda match: match[1] + '\n' + fragment + '\n  ' + match[2],
    page, count=1, flags=re.S)
assert count == 1
page = page.replace(' <span class="soon">Soon</span>', '')
page = page.replace('</head>', '<link rel="stylesheet" href="/ui/finance.css">\n</head>')
page = page.replace('<script>\n/* ============================== shared',
                    '<script src="/ui/finance.js"></script>\n<script>\n/* ============================== shared')
page = page.replace("anomaly: 'Anomaly adapter' };", "anomaly: 'Anomaly adapter', finance_policy: 'Nifty trading adapter' };")
page = page.replace("  if (v === 'chat') $('box').focus();",
                    "  if (v === 'chat') $('box').focus();\n  if (v === 'finance') window.FinanceWorkspace.show();\n  else window.FinanceWorkspace.hide();")
assert 'window.FinanceWorkspace.show()' in page
assert 'src="/ui/finance.js"' in page
assert '<iframe' not in page
(ui / 'index.html').write_text(page.rstrip() + '\n', encoding='utf-8', newline='\r\n')
print('Built Finance inside the original Polyvalent Chat/Anomaly UI')
