"""Assemble plan sections (p*.md) into PLAN.md and a styled HTML page.

Usage: python render_plan.py <sections_dir> <out_md> <out_html>
"""

import glob
import html
import os
import re
import sys

import markdown

SECTIONS_DIR, OUT_MD, OUT_HTML = sys.argv[1], sys.argv[2], sys.argv[3]

parts = []
for path in sorted(glob.glob(os.path.join(SECTIONS_DIR, "p*.md"))):
    with open(path, encoding="utf-8") as fh:
        parts.append(fh.read().strip() + "\n")
md_text = "\n\n".join(parts)
with open(OUT_MD, "w", encoding="utf-8") as fh:
    fh.write(md_text)

md = markdown.Markdown(
    extensions=["tables", "fenced_code", "toc", "sane_lists"],
    extension_configs={"toc": {"toc_depth": "2-3", "permalink": False}},
)
# The plan's formulas use single asterisks (complex conjugates), underscores (subscripts) and
# bracket-parenthesis sequences that Markdown would read as emphasis and links. Bold is written
# with double asterisks only, and the document contains no Markdown links or images, so the
# underscore-emphasis and link patterns are removed and lone asterisks are escaped.
for _name in ("em_strong2", "link", "reference", "short_reference", "image_link",
              "image_reference", "short_image_ref", "autolink", "automail"):
    try:
        md.inlinePatterns.deregister(_name)
    except (KeyError, ValueError):
        pass
md_for_html = re.sub(r"(?<![*\\])\*(?!\*)", r"\\*", md_text)
body = md.convert(md_for_html)

TAGS = {
    "verified": ("v", "verified"),
    "corrected": ("c", "corrected"),
    "extracted": ("x", "extracted"),
    "background": ("b", "background"),
    "recomputed": ("r", "recomputed"),
}


def badge(m):
    key = m.group(1).lower()
    cls, label = TAGS[key]
    qual = (m.group(2) or "").strip(" :,;")
    if qual:
        return f'<span class="tag {cls}">{label}<span class="q"> \u00b7 {html.escape(qual)}</span></span>'
    return f'<span class="tag {cls}">{label}</span>'


TAG_RE = r"\[(verified|corrected|extracted|background|recomputed)((?::|,|;|\s)[^\]]{1,120})?\]"
body = re.sub(r"<strong>" + TAG_RE + r"</strong>", badge, body)
body = re.sub(TAG_RE, badge, body)

# table of contents from h2 headings (ids added by the toc extension)
toc_items = []
for m in re.finditer(r'<h2 id="([^"]+)">(.*?)</h2>', body):
    text = re.sub(r"<[^>]+>", "", m.group(2))
    toc_items.append((m.group(1), html.unescape(text)))
toc_html = "".join(
    f'<li><a href="#{i}">{html.escape(t)}</a></li>' for i, t in toc_items
)

CSS = """
:root{--bg:#F2F5F7;--surface:#FFFFFF;--ink:#16222D;--muted:#566573;--hair:#D3DBE2;--accent:#0B7C8C;--accent-soft:#DDF1F3;--accent-line:#1A9AAB;--amber:#9A6A12;--amber-soft:#F7EBD2;--slate:#5D6C7B;--slate-soft:#E7ECF0;--code-bg:#EBF0F3;color-scheme:light}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0E1418;--surface:#151D24;--ink:#E4EBF1;--muted:#9FB0BE;--hair:#29353F;--accent:#43C6D6;--accent-soft:#0F2E33;--accent-line:#43C6D6;--amber:#E3B25B;--amber-soft:#33270F;--slate:#A3B1BE;--slate-soft:#202A33;--code-bg:#1B252E;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0E1418;--surface:#151D24;--ink:#E4EBF1;--muted:#9FB0BE;--hair:#29353F;--accent:#43C6D6;--accent-soft:#0F2E33;--accent-line:#43C6D6;--amber:#E3B25B;--amber-soft:#33270F;--slate:#A3B1BE;--slate-soft:#202A33;--code-bg:#1B252E;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16.5px/1.6 "STIX Two Text","Times New Roman",Georgia,serif;-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:3px}
a:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
.page{max-width:1160px;margin:0 auto;padding:40px 24px 96px;display:grid;grid-template-columns:250px minmax(0,800px);gap:56px;align-items:start}
@media (max-width:1000px){.page{grid-template-columns:minmax(0,1fr);gap:24px}.toc{position:static !important;border-bottom:1px solid var(--hair);padding-bottom:16px}}
.toc{position:sticky;top:24px;font-family:Archivo,"Helvetica Neue",Arial,sans-serif;font-size:12.5px;line-height:1.4;max-height:calc(100vh - 48px);overflow-y:auto}
.toc .eyebrow{margin-bottom:10px}
.toc ol{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:6px}
.toc a{color:var(--muted);text-decoration:none}
.toc a:hover{color:var(--ink)}
main{min-width:0}
h1,h2,h3,h4{font-family:Archivo,"Helvetica Neue",Arial,sans-serif;text-wrap:balance;line-height:1.15;margin:0}
h1{font-size:38px;font-weight:700;letter-spacing:-0.02em;margin:8px 0 18px}
h2{font-size:25px;font-weight:700;letter-spacing:-0.01em;margin:56px 0 12px;padding-top:22px;border-top:1px solid var(--hair)}
h3{font-size:18.5px;font-weight:600;margin:30px 0 8px}
h4{font-size:14.5px;font-weight:600;margin:22px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
p{margin:0 0 1em}
.eyebrow{font-family:Archivo,"Helvetica Neue",Arial,sans-serif;font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);font-weight:600}
ul,ol{padding-left:1.3em;margin:0 0 1em}
li{margin-bottom:.35em}
li p{margin:0 0 .4em}
strong{font-weight:600}
code{font-family:"IBM Plex Mono",Menlo,Consolas,monospace;font-size:13.5px;background:var(--code-bg);padding:1px 5px;border-radius:3px}
pre{background:var(--code-bg);padding:14px 16px;border-radius:4px;overflow-x:auto;font-size:13px;line-height:1.5;margin:0 0 1.2em}
pre code{background:none;padding:0;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.4;margin:0 0 1.4em;display:block;overflow-x:auto}
th{font-family:Archivo,"Helvetica Neue",Arial,sans-serif;font-size:11.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);text-align:left;padding:8px 10px 8px 0;border-bottom:1px solid var(--ink);font-weight:600;vertical-align:bottom}
td{padding:7px 10px 7px 0;border-bottom:1px solid var(--hair);vertical-align:top;font-variant-numeric:tabular-nums}
blockquote{margin:0 0 1em;padding:2px 0 2px 14px;border-left:2px solid var(--hair);color:var(--muted)}
.tag{display:inline-block;font-family:"IBM Plex Mono",Menlo,Consolas,monospace;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;padding:1px 6px;border-radius:2px;vertical-align:middle;white-space:nowrap}
.tag.v{background:var(--accent-soft);color:var(--accent)}
.tag.c{background:var(--amber-soft);color:var(--amber)}
.tag.x{background:var(--slate-soft);color:var(--slate)}
.tag.b{border:1px dashed var(--muted);color:var(--muted)}
.tag.r{background:var(--slate-soft);color:var(--ink);border:1px solid var(--hair)}
.tag .q{text-transform:none;letter-spacing:0;opacity:.85}
hr{border:0;border-top:1px solid var(--hair);margin:32px 0}
"""

page = f"""<title>qutip-trap Blueprint</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=STIX+Two+Text:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="page">
<nav class="toc" aria-label="Contents"><div class="eyebrow">Contents</div><ol>{toc_html}</ol></nav>
<main>
{body}
</main>
</div>
"""
with open(OUT_HTML, "w", encoding="utf-8") as fh:
    fh.write(page)
print(
    "sections:",
    len(parts),
    "| markdown bytes:",
    len(md_text.encode()),
    "| html bytes:",
    len(page.encode()),
    "| h2 count:",
    len(toc_items),
)
