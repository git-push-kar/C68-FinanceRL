# Fresh GitHub recovery verification — 16 November 2026

Remote tested: `https://github.com/git-push-kar/C68-FinanceRL.git`, branch `main`,
commit `45f69abe2bc234d30d347da4a7f65a243b3874c5`.

A separate clone at `C:/Users/Parth/Documents/C68-FinanceRL-RecoveryCheck`
was created from GitHub with initial LFS smudging disabled. The selected demo
adapter, price panel and canonical research CSV were then downloaded using
`git lfs pull` with explicit include paths. Thus the model and panel used below
came back from the remote, rather than being copied from the working checkout.

- Selected adapter SHA256: `8b678cad4b5b8c100a717deccefd8825e72520c78e8aa823c18dfcdcc0dd0a35`.
- Fresh-clone suite: **35 passed**, one Starlette/httpx deprecation warning.
- A server launched with the fresh clone's `demo` as working directory used the
  clone's application source, selected adapter and panel. The pristine base and
  installed Python dependencies were reused from this PC. **This is not a clean
  dependency installation test or a verification on a second physical device.**
- The new integrity guard verified the pinned full base weight SHA256 on first
  policy request. The actual model ran **54 calls** through **270 sessions**.
- All daily portfolio balances matched the original research record exactly
  (maximum absolute difference **0.0 INR**).
- Final equity **113282.88926029207 INR**, total fees **240.4142855807366 INR**.
- Production browser playback completed the same interval in **42.93 seconds**
  with actual HTTP/model calls, no route mocks and no JavaScript errors. Desktop,
  mobile and actual trade inspection were captured. The browser script's only
  local modification was screenshot positioning, not trading behavior.

Current evidence is under `demo/finance/evaluation/`; the presentation image and
recording are `demo/portfolio-paper-live.png` and `.webm`. The API run and browser
run used separate accounts, both starting from 100000 INR. Their speed describes
this machine's historical sandbox, not real market execution latency.

The complete checkpoint release has a separate recovery receipt in `archives/`.
One historical tensor was independently restored from a local checkpoint ZIP
with SHA256 and size verification, and all 4663 unique archived weight objects
were streamed and independently checked against their content-addressed names.
The metadata ZIP was previously checked against all 66432 manifest entries with
zero differences. These are file-integrity checks, not financial-performance or
scientific-validity certification.
