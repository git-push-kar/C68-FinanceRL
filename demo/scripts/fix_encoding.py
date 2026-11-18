import pathlib
p = pathlib.Path(r"C:\Users\Admin\Desktop\final\deepfake\wrapper.py")
raw = p.read_bytes()
for enc in ("utf-8", "gbk", "cp1252"):
    try:
        text = raw.decode(enc)
        print("decoded as", enc)
        break
    except UnicodeDecodeError:
        continue
p.write_text(text, encoding="utf-8")
print("rewrote utf-8, lines:", len(text.splitlines()))
