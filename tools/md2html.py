#!/usr/bin/env python3
"""把 .md 转成仓库既有的 .html 配套版本。用法: gen.py <md路径...>"""
import re, sys, io, os, html as _html
import markdown

# ---- 复制自既有 HTML 的 <style> 模板（第 6~244 行）----
STYLE = """
    :root {
      --bg:        oklch(97% 0.005 250);
      --surface:   oklch(100% 0 0);
      --border:    oklch(88% 0.01 250);
      --muted:     oklch(55% 0.02 250);
      --text:      oklch(20% 0.02 250);
      --accent:    oklch(52% 0.18 260);
      --accent-bg: oklch(95% 0.04 260);
      --code-bg:   oklch(22% 0.03 250);
      --code-fg:   oklch(88% 0.02 250);
      --link:      oklch(48% 0.18 260);
      --link-v:    oklch(42% 0.14 290);
      --head-bg:   oklch(28% 0.06 260);
      --head-fg:   oklch(97% 0.01 260);
      --th-bg:     oklch(94% 0.04 260);
      --tr-alt:    oklch(96% 0.01 250);
      --blockquote: oklch(92% 0.05 250);
      --star-gold: oklch(72% 0.18 85);
      font-size: 16px;
    }
    *, *::before, *::after { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, "PingFang SC", "Hiragino Sans GB",
                   "Microsoft YaHei", "Segoe UI", system-ui, sans-serif;
      line-height: 1.75;
    }
    .page-wrap {
      max-width: 860px;
      margin: 0 auto;
      padding: 2rem 1.5rem 4rem;
    }
    /* headings */
    h1,h2,h3,h4,h5,h6 {
      font-weight: 700;
      line-height: 1.3;
      margin: 2rem 0 0.6rem;
    }
    h1 { font-size: 1.9rem; color: oklch(30% 0.08 260); border-bottom: 2px solid var(--accent); padding-bottom: 0.3rem; }
    h2 { font-size: 1.45rem; color: oklch(28% 0.06 260); border-left: 4px solid var(--accent); padding-left: 0.6rem; }
    h3 { font-size: 1.15rem; color: oklch(32% 0.05 250); }
    h4 { font-size: 1rem; color: var(--muted); }
    /* links */
    a { color: var(--link); text-decoration: none; }
    a:visited { color: var(--link-v); }
    a:hover { text-decoration: underline; }
    /* paragraph */
    p { margin: 0.6rem 0 0.8rem; }
    /* blockquote */
    blockquote {
      margin: 1rem 0;
      padding: 0.7rem 1rem;
      background: var(--blockquote);
      border-left: 4px solid var(--accent);
      border-radius: 0 6px 6px 0;
      color: oklch(30% 0.04 250);
    }
    blockquote p { margin: 0.2rem 0; }
    /* inline code */
    code {
      font-family: "JetBrains Mono", "Cascadia Code", "SF Mono", "Fira Code",
                   "Source Code Pro", Menlo, Consolas, monospace;
      font-size: 0.88em;
      background: var(--code-bg);
      color: var(--code-fg);
      padding: 0.15em 0.4em;
      border-radius: 4px;
    }
    /* fenced code blocks */
    .codehilite {
      margin: 1rem 0;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 4px oklch(0% 0 0 / 15%);
    }
    .codehilite pre {
      margin: 0;
      padding: 1rem 1.2rem;
      overflow-x: auto;
      font-size: 0.85rem;
      line-height: 1.6;
      background: var(--code-bg) !important;
      color: var(--code-fg) !important;
    }
    .codehilite code { background: none; padding: 0; font-size: inherit; color: inherit; }
    /* plain pre (ASCII diagrams) */
    pre:not(.codehilite pre) {
      background: var(--code-bg);
      color: var(--code-fg);
      padding: 1rem 1.2rem;
      border-radius: 8px;
      overflow-x: auto;
      font-size: 0.82rem;
      line-height: 1.55;
      margin: 1rem 0;
    }
    /* tables */
    .md-body table {
      width: 100%;
      border-collapse: collapse;
      margin: 1.2rem 0;
      font-size: 0.92rem;
      box-shadow: 0 1px 3px oklch(0% 0 0 / 10%);
      border-radius: 8px;
      overflow: hidden;
    }
    .md-body thead tr { background: var(--th-bg); color: oklch(25% 0.07 260); }
    .md-body th { padding: 0.65rem 0.9rem; text-align: left; font-weight: 600; border-bottom: 2px solid var(--border); }
    .md-body td { padding: 0.55rem 0.9rem; border-bottom: 1px solid var(--border); vertical-align: top; }
    .md-body tr:last-child td { border-bottom: none; }
    .md-body tbody tr:nth-child(even) { background: var(--tr-alt); }
    .md-body tbody tr:hover { background: var(--accent-bg); transition: background 0.12s; }
    /* bold inside td */
    td strong { color: oklch(35% 0.12 260); }
    /* ★ markers */
    .md-body p, .md-body li {
      /* keep star chars but add colour */
    }
    /* horizontal rule */
    hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }
    /* lists */
    ul, ol { padding-left: 1.4rem; margin: 0.4rem 0 0.8rem; }
    li { margin: 0.25rem 0; }
    li > ul, li > ol { margin: 0.2rem 0; }
    /* nav breadcrumb at top */
    .nav-bar {
      background: var(--head-bg);
      color: var(--head-fg);
      padding: 0.5rem 1.2rem;
      font-size: 0.82rem;
      border-radius: 8px 8px 0 0;
      margin-bottom: 0;
    }
    .nav-bar a { color: oklch(80% 0.08 250); }
    .nav-bar a:hover { color: #fff; }
    /* TOC */
    .toc { background: var(--accent-bg); border: 1px solid var(--border); border-radius: 8px;
            padding: 0.8rem 1.2rem; margin: 1.2rem 0; font-size: 0.9rem; }
    .toc ul { margin: 0.2rem 0; }
    /* footer */
    .page-footer {
      margin-top: 3rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
      font-size: 0.8rem;
      color: var(--muted);
      text-align: center;
    }
"""

# ---- Pygments one-dark 配色 ----
# 不手抄：直接��� Pygments 导出，保证与仓库既有 HTML 逐字节一致。
# ���验证 HtmlFormatter(style='one-dark').get_style_defs('.codehilite')
# 与仓库既有 .html ���的那 85 行完全相同。
def _pygments_css():
    from pygments.formatters import HtmlFormatter
    return HtmlFormatter(style='one-dark').get_style_defs('.codehilite')

PYG = "    /* Pygments one-dark overrides */\n    " + _pygments_css()

EXTENSIONS = ['extra', 'codehilite', 'tables', 'fenced_code', 'sane_lists',
              'toc', 'nl2br']
EXTENSION_CONFIGS = {
    'codehilite': {'guess_lang': False, 'css_class': 'codehilite'},
}

def md_to_html(fname):
    src = io.open(fname, encoding='utf-8').read()
    body = markdown.markdown(src, extensions=EXTENSIONS,
                             extension_configs=EXTENSION_CONFIGS)
    # 一级标题作为 <title>
    m = re.search(r'^# (.+)$', src, re.M)
    title = m.group(1).strip() if m else os.path.basename(fname)
    title_esc = _html.escape(title, quote=False)  # <title> 是元素内容，" 无需转义
    # 内部 .md 链接 → .html
    body = re.sub(r'(href="[^"]*?)\.md(?=["#])', r'\1.html', body)
    html = (f'<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n'
            f'<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{title_esc}</title>\n'
            f'<style>{STYLE}{PYG}\n</style>\n</head>\n<body>\n'
            f'<div class="page-wrap">\n<div class="md-body">\n'
            f'{body}\n</div>\n'
            f'<div class="page-footer">业界前沿技术报告及论文研究 · 转换自 Markdown</div>\n'
            f'</div>\n</body>\n</html>\n')
    return title, html

if __name__ == '__main__':
    for p in sys.argv[1:]:
        t, h = md_to_html(p)
        out = os.path.splitext(p)[0] + '.html'
        io.open(out, 'w', encoding='utf-8').write(h)
        print(f'写出 {out}  (title={t!r})')
