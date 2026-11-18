import re, pathlib
log = pathlib.Path(r"C:\Users\Admin\AppData\Local\Temp\opencode\real3.log").read_text(
    encoding="utf-8", errors="replace")
lines = log.splitlines()
hit = [i for i, l in enumerate(lines) if "Traceback" in l]
if hit:
    print("\n".join(lines[hit[-1]:hit[-1] + 30]))
else:
    print("NO TRACEBACK; tail:")
    print("\n".join(lines[-15:]))
