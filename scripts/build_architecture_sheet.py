"""Build the matching HTML/PDF architecture summaries (no performance literals).

Optional authoring dependency: reportlab. Not needed for training or Streamlit.
Run: uv run --with reportlab python scripts/build_architecture_sheet.py
"""
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
PAPER = [
    ('Available visits', 'Values + observation masks\nRecorded time gaps\nPublic V=29; simulated V=52'),
    ('Fixed graph + R-GCN', 'Two layers, node width 32\nLatent code: 32 + 16\nObservation-mask / input skip'),
    ('MRTF', 'Past-only rolling windows\n1 / 3 / 6 visits\nGated fusion to width 64'),
    ('Temporal Transformer', 'Two layers, four heads\nPast-only attention + gaps\nTemporal embedding h_t:64'),
    ('Factual path + heads', 'Noise / treatment-action path\nFlare + severity mixture\nThree Gaussian components'),
    ('Paper output', 'Deterministic prediction\nValidation Platt calibration\nFlare probability + severity'),
]
PROPOSED = [
    ('Reuse the Paper network', 'Same frozen neural weights\nPaper margin / severity starting point\nTemporal embedding h_t'),
    ('Add observed history', 'Nine past-only groups per variable\nPlus gap and h_t: 9V + 65 inputs\n326 public / 533 main simulation'),
    ('Temporal Residual Boosting', 'Classification with Paper base margin\nRegression on severity residuals\nValidation selects settings / strengths'),
    ('Proposed output', 'sigmoid(ell_P + alpha * delta)\nclip(s_P + beta * residual, 0, 4)\nOne proposed framework, two task heads'),
]
CARDS = [
    ('Why add the correction?',
     'Compression can leave useful measurements and trends insufficiently expressed. '
     'TRB sees current values, missingness, running / recent history, changes, rates and observation age. '
     'It learns nonlinear corrections without replacing R-GCN, MRTF or the TCT. '
     'XGBoost is established; the contribution is this temporal integration.'),
    ('How the methodology changes',
     'Stage 1: fit the shared neural model with validation early stopping. '
     'Stage 2: freeze it and fit correction heads on training patients. '
     'Validation selects settings and correction strengths; test patients only score the final pair. '
     'Neural seed 73 and split seed 42 are frozen defaults. No future outcome is a prediction input.'),
    ('How to interpret the evidence',
     'Current scores and gains are generated in docs/MEASURED_RESULTS.md and its JSON snapshot. '
     'Read all metrics and bootstrap intervals, including unfavorable changes. '
     'The authors\' original clinical cohort is unavailable. Local gains do not prove superiority '
     'on that cohort, universal improvement, identified treatment effects or clinical validity.'),
]
INK = colors.HexColor('#203938')
TEAL = colors.HexColor('#177c70')
MUTED = colors.HexColor('#536868')
LINE = colors.HexColor('#cddbd8')
WHITE = colors.white


def wrap(text, width, font='Helvetica', size=10):
    lines, current = [], ''
    for word in text.split():
        candidate = (current + ' ' + word).strip()
        if current and stringWidth(candidate, font, size) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + ([current] if current else [])


def text_lines(c, lines, x, top, size=10, leading=14, color=INK, font='Helvetica'):
    c.setFillColor(color)
    c.setFont(font, size)
    for i, line in enumerate(lines):
        c.drawString(x, top - i * leading, line)
    return top - len(lines) * leading


def arrow(c, x1, x2, y):
    c.setStrokeColor(TEAL)
    c.setLineWidth(1.25)
    c.line(x1, y, x2 - 4, y)
    p = c.beginPath()
    p.moveTo(x2 - 5, y + 3)
    p.lineTo(x2, y)
    p.lineTo(x2 - 5, y - 3)
    p.close()
    c.setFillColor(TEAL)
    c.drawPath(p, fill=1, stroke=0)


def rail(c, title, caption, boxes, bottom, height, fill):
    margin, page_width = 42, landscape(A3)[0]
    width = page_width - 2 * margin
    c.setFillColor(fill)
    c.setStrokeColor(LINE)
    c.roundRect(margin, bottom, width, height, 10, fill=1, stroke=1)
    text_lines(c, [title], margin + 14, bottom + height - 25, 16, font='Helvetica-Bold')
    text_lines(c, [caption], margin + 14, bottom + height - 44, 9.6, color=MUTED)
    gap, inner = 17, width - 28
    box_width = (inner - gap * (len(boxes) - 1)) / len(boxes)
    y, box_height = bottom + 18, 77
    for i, (heading, body) in enumerate(boxes):
        x = margin + 14 + i * (box_width + gap)
        c.setFillColor(WHITE)
        c.setStrokeColor(LINE)
        c.roundRect(x, y, box_width, box_height, 5, fill=1, stroke=1)
        heading_lines = wrap(heading, box_width - 16, 'Helvetica-Bold', 10)
        top = text_lines(c, heading_lines, x + 8, y + box_height - 15, 10, 12, font='Helvetica-Bold')
        body_lines = []
        for line in body.split('\n'):
            body_lines.extend(wrap(line, box_width - 16, size=8.5))
        end = text_lines(c, body_lines, x + 8, top - 3, 8.5, 11, MUTED)
        assert end >= y + 2, (heading, 'overflow')
        if i + 1 < len(boxes):
            arrow(c, x + box_width + 2, x + box_width + gap - 2, y + box_height / 2)


def build_pdf(destination):
    width, height = landscape(A3)
    c = canvas.Canvas(str(destination), pagesize=(width, height))
    c.setTitle('Paper and Proposed frameworks - architecture')
    c.setAuthor('Longitudinal Health Predictor project')
    c.setFillColor(colors.HexColor('#f8faf9'))
    c.rect(0, 0, width, height, fill=1, stroke=0)
    text_lines(c, ['Two frameworks. One shared backbone.'], 42, height - 49, 26, font='Helvetica-Bold')
    text_lines(c, ['Paper-inspired graph / temporal learning, extended with Temporal Residual Boosting (TRB).'],
               43, height - 73, 12, color=MUTED)
    rail(c, '1  PAPER FRAMEWORK - local implementation',
         'Shared backbone for both methods. Graph, masks and transforms fit training data; future visits are excluded.',
         PAPER, 579, 153, colors.HexColor('#eaf1f1'))
    rail(c, '2  PROPOSED FRAMEWORK - the additional stage',
         'The neural architecture stays shared; the complete predictor gains explicit history inputs and a second fitting stage.',
         PROPOSED, 407, 153, colors.HexColor('#e5f3ee'))
    gap, total = 16, width - 84
    card_width, bottom, card_height = (total - 2 * gap) / 3, 171, 217
    for i, (heading, body) in enumerate(CARDS):
        x = 42 + i * (card_width + gap)
        c.setFillColor(WHITE)
        c.setStrokeColor(LINE)
        c.roundRect(x, bottom, card_width, card_height, 9, fill=1, stroke=1)
        text_lines(c, [heading], x + 15, bottom + card_height - 26, 13, font='Helvetica-Bold')
        lines = wrap(body, card_width - 30, size=11)
        end = text_lines(c, lines, x + 15, bottom + card_height - 49, 11, 16, MUTED)
        assert end >= bottom + 10, (heading, 'overflow')
    text_lines(c, ['Paper source: Sharma & Kaur (2025), IEEE Access 13, 150010-150027. DOI: 10.1109/ACCESS.2025.3598276.'],
               43, 140, 10)
    text_lines(c, ['Publication: Sections III-IV and Algorithm 1. Local adaptations and mathematical details: docs/METHODOLOGY.md and CAUSAL_MODEL.md.'],
               43, 122, 9.5, color=MUTED)
    text_lines(c, ['LECG is disabled in both active backbones. The two correction task heads are part of one Proposed framework.'],
               43, 104, 9.5, color=MUTED)
    text_lines(c, ['This sheet contains architecture, not fixed performance numbers. See MEASURED_RESULTS.md for the latest verified comparison.'],
               43, 79, 10, font='Helvetica-Bold')
    text_lines(c, ['Academic research prototype. Temporal masking and causal terminology do not establish clinical causal identification.'],
               43, 58, 9, color=MUTED)
    c.showPage()
    c.save()


def build_html(destination):
    def boxes(items):
        return '<div class="flow">' + ''.join(
            '<div class="box"><h3>' + escape(title) + '</h3><p>' + escape(body).replace('\n', '<br>') + '</p></div>'
            for title, body in items) + '</div>'
    cards = ''.join('<section class="card"><h2>' + escape(title) + '</h2><p>' + escape(body) + '</p></section>'
                    for title, body in CARDS)
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paper and Proposed frameworks - architecture</title><style>
*{box-sizing:border-box}body{margin:0;background:#f8faf9;color:#203938;font:15px/1.65 system-ui,sans-serif}
main{max-width:1520px;margin:auto;padding:40px}h1{font-size:36px;line-height:1.1;letter-spacing:-1px;margin:0 0 12px}
.intro{color:#536868;margin:0 0 28px}h2{font-size:20px;line-height:1.3;margin:0 0 8px}h3{font-size:14px;line-height:1.35;margin:0 0 8px}
.rail{border:1px solid #cddbd8;border-radius:12px;background:#eaf1f1;padding:20px;margin-bottom:20px}.proposed{background:#e5f3ee}
.caption{font-size:13px;color:#536868;margin:0 0 20px}.flow{display:flex;gap:17px}.box{position:relative;flex:1;background:white;border:1px solid #cddbd8;border-radius:6px;padding:14px 12px}
.box p{font-size:12px;line-height:1.5;margin:0}.box:not(:last-child):after{content:'\u2192';position:absolute;right:-17px;top:42%;width:17px;text-align:center;color:#177c70}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}.card{background:white;border:1px solid #cddbd8;border-radius:10px;padding:20px}.card p{color:#536868;margin-bottom:0;font-size:14px}
footer{margin-top:28px;border-top:1px solid #cddbd8;padding-top:16px;color:#536868;font-size:13px}a{color:#177c70}
@media(max-width:1050px){main{padding:24px}.flow{flex-direction:column}.box:not(:last-child):after{content:'\u2193';top:auto;bottom:-20px;right:50%}.cards{grid-template-columns:1fr}}
@media print{@page{size:A3 landscape;margin:12mm}main{padding:0}body{font-size:12px}h1{font-size:28px}.rail{break-inside:avoid}.cards{gap:12px}.card{padding:14px}a{color:inherit;text-decoration:none}}
</style></head><body><main><h1>Two frameworks. One shared backbone.</h1>
<p class="intro">Paper-inspired graph / temporal learning, extended with Temporal Residual Boosting (TRB).</p>
<section class="rail"><h2>1 &nbsp; Paper framework - local implementation</h2>
<p class="caption">Shared backbone for both methods. Training-only transforms and graph construction; future visits excluded.</p>'''
    html += boxes(PAPER) + '</section><section class="rail proposed"><h2>2 &nbsp; Proposed framework - the additional stage</h2>'
    html += '<p class="caption">The complete predictor gains explicit history inputs and a second fitting stage; neural weights stay shared.</p>'
    html += boxes(PROPOSED) + '</section><div class="cards">' + cards + '</div>'
    html += '''<footer><p>Source: Sharma &amp; Kaur (2025), <i>IEEE Access</i> 13, 150010-150027.
<a href="https://doi.org/10.1109/ACCESS.2025.3598276">DOI</a> · <a href="paper/reference.pdf">Supplied paper</a>.</p>
<p>LECG is disabled in both active backbones. The two correction heads belong to one Proposed framework.</p>
<p><a href="METHODOLOGY.md">Methodology and equations</a> · <a href="CAUSAL_MODEL.md">Implementation mapping</a> ·
<a href="MEASURED_RESULTS.md">Latest measured improvement</a> · <a href="architecture_sheet.pdf">PDF</a></p>
<p>Academic research prototype. Architecture alone does not establish causal identification or clinical validity.</p></footer></main></body></html>'''
    destination.write_text(html, encoding='utf-8')


if __name__ == '__main__':
    build_html(ROOT / 'docs/architecture_sheet.html')
    build_pdf(ROOT / 'docs/architecture_sheet.pdf')
    print('Rebuilt architecture_sheet.html and architecture_sheet.pdf; render the PDF to architecture_sheet.png for visual review.')
