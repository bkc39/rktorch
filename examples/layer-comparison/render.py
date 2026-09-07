"""Render matching layer definitions from the three executable source files."""

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMPONENTS = [
    ("Projection", "projection"),
    ("ChannelNorm", "channel_norm"),
    ("ResidualBlock", "residual_block"),
    ("ResidualStage", "residual_stage"),
    ("SmallResNet", "small_resnet"),
    ("CausalSelfAttention", "causal_self_attention"),
    ("FeedForward", "feed_forward"),
    ("TransformerBlock", "transformer_block"),
    ("TransformerStack", "transformer_stack"),
]


def extract(source, pattern):
    match = re.search(pattern, source, re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError(f"Definition missing: {pattern}")
    return html.escape(match.group(0).strip())


def main():
    racket = (ROOT / "models.rkt").read_text()
    python = (ROOT / "models.py").read_text()
    ocaml = (ROOT / "models.ml").read_text()
    ocaml_names = "|".join(name for _, name in COMPONENTS) + "|demo"
    sections = []
    for name, ocaml_name in COMPONENTS:
        snippets = [
            extract(racket, rf"^\(define-layer {name}\b.*?(?=^\(define-layer |\Z)"),
            extract(python, rf"^class {name}\b.*?(?=^class |^if __name__|\Z)"),
            extract(ocaml, rf"^let {ocaml_name}\b.*?(?=^let (?:{ocaml_names})\b|\Z)"),
        ]
        columns = "".join(
            f"<article><h3>{language}</h3><pre><code>{snippet}</code></pre></article>"
            for language, snippet in zip(
                ["Racket · define-layer", "Python · nn.Module", "OCaml · Layer.of_fn"],
                snippets,
            )
        )
        sections.append(
            f'<section id="{name}"><h2>{name}</h2><div class="columns">{columns}</div></section>'
        )
    navigation = " · ".join(f'<a href="#{name}">{name}</a>' for name, _ in COMPONENTS)
    page = """<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Layer composition: Racket / Python / OCaml</title>
<style>
body{margin:0;background:#f6f5f1;color:#222;font:15px/1.5 system-ui,sans-serif}
header{padding:24px 30px;border-bottom:1px solid #ccc;background:white}
h1{margin:0;font-size:28px}header p{max-width:1050px}nav{line-height:2}
a{color:#205ba0}section{padding:18px 22px;scroll-margin-top:10px}
h2{font-size:22px;margin:8px 0}
.columns{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
article{background:white;border:1px solid #d7d5ce;border-radius:6px;min-width:0}
h3{font-size:14px;margin:0;padding:10px 14px;border-bottom:1px solid #ddd;background:#eceeea}
pre{font:12px/1.55 ui-monospace,Menlo,Consolas,monospace;white-space:pre;overflow:auto;padding:14px;margin:0}
@media(max-width:1000px){.columns{grid-template-columns:1fr}pre{font-size:13px}}
</style>
<header><h1>Layer composition in three languages</h1>
<p>Read each component across the columns. Racket uses threading and T for
rank-two weights. Batched attention retains explicit axis swaps.</p>
<p>The CNN uses channel-wise LayerNorm and biased convolutions; it is a
ResNet-style model, not canonical ResNet-18. procedure-&gt;Layer is not yet a
public Racket API. These examples use the supported define-layer syntax.</p>
<p><a href="README.md">Architecture, API status, and run commands</a> ·
Full source: <a href="models.rkt">Racket</a> /
<a href="models.py">Python</a> / <a href="models.ml">OCaml</a></p>
"""
    page += f"<nav>{navigation}</nav></header>" + "".join(sections) + "</html>\n"
    (ROOT / "side-by-side.html").write_text(page)


if __name__ == "__main__":
    main()
