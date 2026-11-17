# Finance in the existing Polyvalent UI — 17 November 2026

The user clarified that Finance should follow the existing chatbot/anomaly UI.
The reference was located in [git-push-kar/Polyvalent](https://github.com/git-push-kar/Polyvalent),
`ui/index.html`, Git blob `4fad71c9c8d645a5acc633efef9e50a9407aaa4b`.
It has Chat, Anomaly detection and an unfinished Finance tab. The previous Finance
demo already reused that outer shell, but embedded a separately styled portfolio
page. That embedded page has now been replaced with a native Finance workspace.

## Implementation

- The original Chat and Anomaly section markup is preserved verbatim. The shared
  navigation, adapter-status pill, theme variables, cards and controls are reused.
- Finance follows the Anomaly layout: a simulation/account sidebar, main metrics,
  a portfolio chart, holdings, decision records and activity. It uses the same
  purple accent and automatic light/dark theme.
- Finance runs in the same document. All its element IDs have `fin-` prefixes,
  CSS is scoped to its workspace, and its JavaScript state is isolated in a closure.
  No iframe, second app shell or Finance chatbot is used.
- The account initializes on first opening Finance. Leaving the tab stops
  playback; an already submitted step may finish. Returning preserves the account.
- Old `/ui/paper.html` and `/ui/finance.html` links redirect to `/#finance`.
- Removed a duplicated Finance router registration noticed during integration.
  No changes were made to allocation mathematics, weights, fees or evaluation data.

`demo/ui/reference/polyvalent-original-index.html` preserves the upstream source.
`demo/ui/finance-workspace.html` is the Finance fragment, `finance.css` uses shared
theme variables, and `finance.js` contains the actual paper-account controller.
Run `python scripts/build_finance_workspace.py` from `demo/` to rebuild the main
page from the upstream reference and Finance fragment. Edit those inputs rather
than discarding Chat/Anomaly changes through an unrelated new page.

## Verification

- Existing Python suite: **35 passed**, one Starlette/httpx deprecation warning.
- UI transport fixtures: next day/rebalance, trade selection, holdings search,
  benchmark visibility, CSV export, completion and mobile width passed. Those
  fixtures are explicitly not model-quality evidence.
- `scripts/verify_finance_native_ui.py` used real production HTTP calls and
  **two actual adapter calls**. It checked preserved Chat/Anomaly markup, tab
  navigation, lazy initialization, absence of iframes/duplicate IDs, playback
  stopping on tab changes, matching primary colours, mobile width, both themes,
  legacy redirects and absence of JavaScript errors.
- The final native UI completed **270 sessions / 54 actual adapter calls** in
  **41.69 seconds**. Every daily balance matched the original research record:
  maximum absolute difference **0.0 INR**. Ending value **113282.88926029207 INR**.
- Raw native account output: `demo/finance/evaluation/paper_native_live_run.json`.
  UI receipts: `demo/finance/evaluation/paper_ui/`. The reviewed real-model image
  and recording are `demo/portfolio-paper-live.png` and `.webm`.

No Finance retraining was performed. Performance and publication limitations in
the main research report are unchanged. Chat/Anomaly navigation checks do not
certify the quality of their unavailable trained models. Finance generated
conversation remains removed after its earlier factual audit; the decision
inspector shows allocation and accounting records, not generated justification.

## Current use

Open http://127.0.0.1:8765/#finance and choose **Start simulation**. Use the existing
Chat and Anomaly tabs in the same header. The server now serves the authoritative
`C:/Users/Parth/Documents/C68-FinanceRL/demo` source; it no longer serves the
temporary recovery clone. The selected policy and pinned base are unchanged.
