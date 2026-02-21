from __future__ import annotations

from html import escape
from string import Template

_REPORT_DOCUMENT_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>$title</title>
<link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css' rel='stylesheet'>
<link rel='preconnect' href='https://fonts.googleapis.com'>
<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>
<link href='https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap' rel='stylesheet'>
<style>$styles</style>
</head>
<body>
<main class='report-shell' data-report-root='true'>
<header class='report-hero'>
<h1 class='report-title'>$title</h1>
$intro_html
</header>
<section class='report-actions'>
<button type='button' class='btn btn-sm btn-outline-secondary action-button' data-expand-all='true'>Expand all</button>
<button type='button' class='btn btn-sm btn-outline-secondary action-button' data-collapse-all='true'>Collapse all</button>
</section>
$body_html
</main>
<script>$script</script>
</body>
</html>"""
)

_REPORT_STYLES = """
:root{
  --bg-top:#f4f8f6;
  --bg-mid:#ecf5f4;
  --bg-bottom:#f8fbfa;
  --panel-bg:#ffffff;
  --panel-border:#d1ddd8;
  --panel-shadow:0 14px 35px rgba(15, 23, 42, .08);
  --text-main:#122027;
  --text-muted:#45606c;
  --heading:#14252d;
  --hero-accent-a:#0d9488;
  --hero-accent-b:#f59e0b;
  --header-bg:#eef4f2;
  --header-bg-strong:#e2eeeb;
  --radius-md:10px;
  --radius-lg:12px;
  --space-block:.78rem;
}

*,
*::before,
*::after{
  box-sizing:border-box;
}

body{
  margin:0;
  color:var(--text-main);
  line-height:1.55;
  font-family:"Manrope","Avenir Next","Segoe UI",sans-serif;
  background:
    radial-gradient(1100px 500px at 5% -20%, rgba(13,148,136,.16), transparent 58%),
    radial-gradient(900px 450px at 96% -10%, rgba(245,158,11,.14), transparent 54%),
    linear-gradient(180deg, var(--bg-top) 0%, var(--bg-mid) 48%, var(--bg-bottom) 100%);
}

.report-shell{
  max-width:1120px;
  margin:1.5rem auto 2.1rem auto;
  padding:0 1rem;
}

.report-hero{
  position:relative;
  background:var(--panel-bg);
  color:var(--heading);
  border:1px solid var(--panel-border);
  border-radius:18px;
  padding:1.15rem 1.2rem;
  box-shadow:var(--panel-shadow);
  overflow:hidden;
  animation:report-fade-up .45s ease both;
}

.report-hero::after{
  content:"";
  position:absolute;
  inset:0 0 auto 0;
  height:6px;
  background:linear-gradient(90deg, var(--hero-accent-a), var(--hero-accent-b));
}

.report-title{
  margin:0;
  color:var(--heading);
  letter-spacing:.01em;
  font-size:clamp(1.3rem, 2.1vw, 1.7rem);
  line-height:1.2;
  font-family:"Fraunces","Iowan Old Style","Palatino Linotype",serif;
  font-weight:700;
}

.report-subtitle{
  margin:.42rem 0 0 0;
  color:var(--text-muted);
  font-size:.97rem;
}

.report-actions{
  margin:.8rem 0 .7rem 0;
  display:flex;
  gap:.45rem;
  flex-wrap:wrap;
}

.action-button{
  border-color:#b7c6c1;
  color:#27414c;
  background:#f9fcfb;
}

.action-button:hover{
  background:#edf5f3;
  border-color:#93a8a1;
  color:#1a3038;
}

.summary-table-wrap{
  margin:var(--space-block) 0 1rem 0;
}

.summary-kv{
  margin:0;
  padding:0;
  display:grid;
  grid-template-columns:minmax(160px, 240px) 1fr;
  gap:.38rem .95rem;
}

.summary-kv .kv-row{
  display:contents;
}

.summary-kv .kv-label{
  margin:0;
  font-size:.86rem;
  text-transform:uppercase;
  letter-spacing:.035em;
  color:#4d6570;
  font-weight:650;
}

.summary-kv .kv-value{
  margin:0;
  color:#12262f;
  min-width:0;
}

.summary-kv .kv-value pre{
  margin:0;
}

.stat-strip{
  display:grid;
  grid-template-columns:repeat(4, minmax(0, 1fr));
  gap:.52rem;
  margin:.3rem 0 var(--space-block) 0;
}

.stat-card{
  border:1px solid #cfdbd7;
  border-radius:var(--radius-md);
  background:#f9fcfb;
  padding:.55rem .62rem;
}

.stat-card .stat-label{
  display:block;
  color:#526975;
  font-size:.76rem;
  letter-spacing:.038em;
  text-transform:uppercase;
  margin-bottom:.1rem;
}

.stat-card .stat-value{
  color:#10222b;
  font-size:1rem;
  font-weight:700;
}

.stat-card.status-fail{
  border-color:#f1c3c3;
  background:#fff6f6;
}

.stat-card.status-warn{
  border-color:#f3dfad;
  background:#fffaf1;
}

.issue-list{
  display:grid;
  gap:.44rem;
}

.issue-head,
.issue-row{
  display:grid;
  grid-template-columns:minmax(112px, 150px) minmax(170px, 260px) 1fr;
  gap:.65rem;
  align-items:start;
}

.issue-head{
  padding:0 .2rem;
}

.issue-head span{
  color:#4e6671;
  font-size:.73rem;
  letter-spacing:.038em;
  text-transform:uppercase;
  font-weight:700;
}

.issue-card{
  border:1px solid #d5e1dd;
  border-radius:var(--radius-md);
  padding:.58rem .68rem;
  background:#fdfefe;
  box-shadow:0 1px 0 rgba(15, 23, 42, .03);
}

.issue-cell{
  min-width:0;
}

.issue-status{
  display:flex;
  align-items:center;
  align-self:center;
}

.issue-scope{
  margin:0 0 .2rem 0;
  font-weight:700;
  color:#20353f;
}

.issue-convention{
  margin:0;
  color:#526671;
  font-size:.84rem;
}

.issue-detail{
  color:#142932;
  margin:0;
  white-space:pre-wrap;
  line-height:1.45;
}

.issue-empty{
  margin:0;
  color:#4b636d;
}

.check-summary-block{
  margin:0;
}

.report-section{
  background:var(--panel-bg);
  border:1px solid var(--panel-border);
  border-radius:var(--radius-lg);
  margin:.65rem 0;
  overflow:hidden;
  box-shadow:0 7px 18px rgba(15, 23, 42, .05);
  animation:report-fade-up .45s ease both;
}

.report-section summary{
  padding:.82rem .95rem;
  cursor:pointer;
  list-style:none;
  font-weight:600;
  display:flex;
  gap:.75rem;
  align-items:center;
  justify-content:space-between;
  background:var(--header-bg);
  color:var(--heading);
}

.report-section summary::-webkit-details-marker{
  display:none;
}

.report-section .summary-badge{
  display:inline-flex;
  align-items:center;
}

.report-badge{
  font-size:.78rem;
  font-weight:700;
  letter-spacing:.02em;
  padding:.42rem .7rem;
}

.report-section .section-title{
  color:var(--heading);
  font-size:.97rem;
  font-weight:680;
  letter-spacing:.01em;
}

.report-section .section-body{
  padding:.86rem .95rem 1rem .95rem;
  border-top:1px solid #dee8e4;
}

.report-section.static-section .section-header{
  padding:.82rem .95rem;
  font-weight:600;
  background:var(--header-bg);
  color:var(--heading);
  display:flex;
  align-items:center;
}

.report-section.static-section .section-body{
  border-top:1px solid #dee8e4;
}

.report-table{
  margin:0;
}

.report-table thead th{
  font-size:.79rem;
  text-transform:uppercase;
  letter-spacing:.04em;
  color:#314551;
  background:#f3f7f6;
  border-bottom:1px solid #cad8d3;
}

.report-table td{
  font-size:.94rem;
  color:#152730;
  vertical-align:top;
  border-color:#dee8e4;
}

.summary-table thead th{
  font-size:.76rem;
}

.variable-report > summary{
  background:var(--header-bg-strong);
}

.section-stack{
  margin-top:var(--space-block);
  display:grid;
  gap:.02rem;
}

.report-section pre{
  background:#f5faf8;
  border:1px solid #d5e2dd;
  border-radius:8px;
  padding:.64rem .7rem;
  margin:0;
  color:#22353f;
  font-size:.86rem;
  white-space:pre-wrap;
}

@keyframes report-fade-up{
  from{opacity:0;transform:translateY(8px);}
  to{opacity:1;transform:translateY(0);}
}

@media (max-width: 768px){
  .report-shell{
    margin:1rem auto 1.5rem auto;
    padding:0 .72rem;
  }

  .report-hero{
    border-radius:14px;
    padding:1rem .92rem;
  }

  .report-section{
    border-radius:var(--radius-md);
    margin:.56rem 0;
  }

  .report-section summary{
    padding:.72rem .82rem;
  }

  .report-section .section-body{
    padding:.74rem .82rem .9rem .82rem;
  }

  .summary-table td:first-child{
    width:auto;
  }

  .summary-kv{
    grid-template-columns:1fr;
    gap:.16rem;
  }

  .summary-kv .kv-label{
    font-size:.74rem;
  }

  .summary-kv .kv-value{
    margin-bottom:.33rem;
  }

  .stat-strip{
    grid-template-columns:repeat(2, minmax(0, 1fr));
  }

  .issue-head{
    display:none;
  }

  .issue-row{
    grid-template-columns:1fr;
    gap:.38rem;
  }
}

@media (prefers-reduced-motion: reduce){
  .report-hero,
  .report-section{
    animation:none;
  }
}
"""

_REPORT_SCRIPT = """
(function () {
  var root = document.querySelector('[data-report-root]');
  if (!root) return;

  var details = root.querySelectorAll('details.report-section');
  var expandBtn = root.querySelector('[data-expand-all]');
  var collapseBtn = root.querySelector('[data-collapse-all]');

  if (!details.length) {
    if (expandBtn) expandBtn.style.display = 'none';
    if (collapseBtn) collapseBtn.style.display = 'none';
    return;
  }

  if (expandBtn) {
    expandBtn.addEventListener('click', function () {
      details.forEach(function (el) { el.open = true; });
    });
  }

  if (collapseBtn) {
    collapseBtn.addEventListener('click', function () {
      details.forEach(function (el) { el.open = false; });
    });
  }
})();
"""


def render_report_document(title: str, intro_html: str, body_html: str) -> str:
    return _REPORT_DOCUMENT_TEMPLATE.substitute(
        title=escape(title),
        intro_html=intro_html,
        body_html=body_html,
        styles=_REPORT_STYLES,
        script=_REPORT_SCRIPT,
    )
