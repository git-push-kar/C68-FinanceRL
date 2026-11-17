"""Real UI/API smoke checks for the existing shell and native Finance tab.

Requires the actual server on 8765 and Playwright/Chromium. No request mocks.
Does not assess the unavailable other-domain models' generation quality.
"""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
out = root / 'finance/evaluation/paper_ui'
out.mkdir(exist_ok=True)
original = (root / 'ui/reference/polyvalent-original-index.html').read_text(encoding='utf-8')
current = (root / 'ui/index.html').read_text(encoding='utf-8')
for name, following in [('chat', 'anomaly'), ('anomaly', 'finance')]:
    begin = '<section class="view" id="view-' + name + '"'
    end = '<section class="view" id="view-' + following + '"'
    assert original[original.index(begin):original.index(end)] == current[current.index(begin):current.index(end)]

errors, finance_requests = [], []
with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    context = browser.new_context(viewport={'width':1440, 'height':1100}, color_scheme='light')
    page = context.new_page()
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.on('request', lambda r: finance_requests.append(r.url) if '/finance/paper/' in r.url else None)
    page.goto('http://127.0.0.1:8765/#chat')
    assert page.locator('#box').is_visible()
    assert not finance_requests, 'Finance must initialize only when its tab is opened'
    assert page.locator('iframe').count() == 0
    duplicate_ids = page.evaluate("""() => {const ids=[...document.querySelectorAll('[id]')].map(e=>e.id); return ids.filter((id,i)=>ids.indexOf(id)!==i)}""")
    assert not duplicate_ids, duplicate_ids
    page.locator('#tab-anomaly').click()
    page.locator('#scenario option').first.wait_for(state='attached')
    assert page.locator('#playBtn').is_visible()
    page.locator('#tab-finance').click()
    page.get_by_text('Ready · 270 historical trading sessions', exact=True).wait_for()
    primary = lambda selector: page.locator(selector).evaluate('(e)=>getComputedStyle(e).backgroundColor')
    assert primary('#fin-play') == primary('#playBtn')
    page.locator('#fin-day').click()
    page.get_by_text('1 / 270 sessions · 4 sessions to rebalance', exact=True).wait_for(timeout=180000)
    page.locator('#fin-rebalance').click()
    page.get_by_text('6 / 270 sessions · 4 sessions to rebalance', exact=True).wait_for()
    page.locator('#fin-speed').select_option('1500')
    page.locator('#fin-play').click()
    page.get_by_text('7 / 270 sessions · 3 sessions to rebalance', exact=True).wait_for()
    page.locator('#tab-chat').click()
    key = page.evaluate("localStorage.getItem('finance-paper-session')")
    endpoint = 'http://127.0.0.1:8765/finance/paper/sessions/' + key
    before = context.request.get(endpoint).json()['sessions']
    page.wait_for_timeout(1800)  # Longer than the selected playback interval.
    after = context.request.get(endpoint).json()['sessions']
    assert before == after, 'Playback continued after leaving Finance'
    assert page.locator('#box').is_visible()
    page.locator('#tab-finance').click()
    assert page.locator('#fin-play').inner_text().strip() == '▶ Resume'
    state = context.request.get(endpoint).json()
    assert all(a['model_receipt']['mode'] == 'real_model' for a in state['activities'])
    for scheme, size in [('light',1440), ('dark',1440), ('dark',390), ('light',390)]:
        page.emulate_media(color_scheme=scheme)
        page.set_viewport_size({'width':size, 'height':1100 if size==1440 else 844})
        assert primary('#fin-play') == primary('#playBtn')
        dimensions = page.locator('body').evaluate('(e)=>({width:e.clientWidth,scroll:e.scrollWidth})')
        assert dimensions['scroll'] <= dimensions['width']+1, dimensions
        page.screenshot(path=str(out / f'native_{scheme}_{size}_actual.png'))
    for legacy in ['paper', 'finance']:
        response = context.request.get('http://127.0.0.1:8765/ui/'+legacy+'.html', max_redirects=0)
        assert response.status == 307 and response.headers['location'] == '/#finance'
    assert not errors, errors
    context.close()
    browser.close()

result = {'mode':'Real production UI/API; no request mocks',
          'reference':'git-push-kar/Polyvalent ui/index.html blob 4fad71c9c8d645a5acc633efef9e50a9407aaa4b',
          'checks':['original Chat and Anomaly section markup preserved','lazy Finance initialization',
                    'native tab without iframe','no duplicate element IDs','existing Chat/Anomaly tab navigation',
                    'real-model next-day and next-rebalance','playback stops on tab change',
                    'shared primary colour in light and dark modes','desktop/mobile root has no horizontal overflow',
                    'old Finance URLs redirect to the native tab','no JavaScript errors'],
          'sessions_in_control_test':after,'actual_adapter_calls':len(state['activities']),
          'javascript_errors':errors,
          'limits':'Tab navigation/markup only for Chat and Anomaly; no other-domain model-quality certification.'}
(out / 'native_integration_verification.json').write_text(json.dumps(result, indent=2))
print('Native Finance integration checks passed; real model calls:', len(state['activities']))
