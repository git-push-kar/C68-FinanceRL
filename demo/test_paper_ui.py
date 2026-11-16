import json,copy
from pathlib import Path
import numpy as np
from playwright.sync_api import sync_playwright
from finance.paper_engine import Panel,PaperAccount
D=Path.cwd();O=D/'finance/evaluation/paper_ui';O.mkdir(exist_ok=True);p=Panel();a=PaperAccount(p);probs=np.asarray(p.manifest['recorded_probabilities']['seed42_64']);snaps={0:a.snapshot()};real=json.loads((D/'finance/evaluation/paper_full_live_run.json').read_text())
for _ in range(270):
 idx=len(a.activities);receipt=real['activities'][min(idx,53)]['model_receipt'];a.step(lambda:(probs,receipt))
 if a.n in [1,6,270]:snaps[a.n]=copy.deepcopy(a.snapshot())
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True);page=browser.new_page(viewport={'width':1440,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://127.0.0.1:8765/#finance');frame=page.frame_locator('iframe[title="Finance portfolio workspace"]');frame.get_by_text('Your paper portfolio',exact=True).wait_for();frame.get_by_role('button',name='Next day',exact=True).wait_for();page.wait_for_timeout(500)
 assert frame.get_by_text('Ready · 270 historical trading sessions',exact=True).is_visible()
 # UI-only transport fixture. Actual model/backend parity is tested separately.
 def response(route):
  body=route.request.post_data_json;n=body['expected_sessions'];target=1 if n==0 else 6 if n==1 else 270;key=route.request.url.split('/sessions/')[1].split('/')[0];route.fulfill(json={'session_id':key,**snaps[target]})
 page.route('**/finance/paper/sessions/*/advance',response)
 frame.get_by_role('button',name='Next day',exact=True).click();frame.get_by_text('1 / 270 sessions · 4 sessions to rebalance',exact=True).wait_for();assert frame.get_by_text('2024-12-31 · Initial allocation',exact=True).is_visible();assert frame.get_by_text('₹1,00,223.60',exact=True).is_visible()
 frame.get_by_role('button',name='View decision',exact=True).click();assert frame.get_by_text('Learned strategy mix',exact=True).is_visible();assert frame.get_by_text('32.88%',exact=True).is_visible()
 frame.get_by_role('button',name='Next rebalance',exact=True).click();frame.get_by_text('6 / 270 sessions · 4 sessions to rebalance',exact=True).wait_for();assert frame.get_by_text('2025-01-07 · Rebalance',exact=True).is_visible()
 frame.get_by_role('textbox',name='Find a holding').fill('HDFCBANK');assert frame.locator('#holdings tr').count()==1;frame.get_by_role('textbox',name='Find a holding').fill('')
 frame.get_by_label('Show benchmarks').uncheck();assert frame.locator('#chart path').count()==1;frame.get_by_label('Show benchmarks').check();assert frame.locator('#chart path').count()==3
 with page.expect_download() as download:frame.get_by_role('button',name='Export activity',exact=True).click()
 download.value.save_as(O/'activity_ui_fixture.csv')
 frame.get_by_role('button',name='Next rebalance',exact=True).click();frame.get_by_text('Completed 270 trading sessions',exact=True).wait_for();assert frame.get_by_role('button',name='Simulation complete',exact=True).is_disabled()
 assert not errors,errors
 page.screenshot(path=str(O/'desktop_fixture.png'),full_page=True)
 page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200);overflow=frame.locator('body').evaluate('(e)=>({scroll:e.scrollWidth,width:e.clientWidth})');assert overflow['scroll']<=overflow['width']+1,overflow
 page.screenshot(path=str(O/'mobile_fixture.png'),full_page=True);browser.close()
 (O/'verification.json').write_text(json.dumps({'mode':'Headless UI transport fixtures using recorded real-model results; not a new model-quality evaluation','checks':['start state','next day','next rebalance','specific activity selection','holdings filtering','benchmark toggle','activity CSV export','end state','mobile horizontal layout','no page JS errors'],'javascript_errors':errors,'mobile_dimensions':overflow},indent=2));print('UI checks passed')
