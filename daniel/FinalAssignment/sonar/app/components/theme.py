"""
SONAR Design System — centralised theme, CSS, and reusable UI helpers.

Every page imports `inject_global_css()` once at the top and `render_footer()`
at the bottom.  Colours, spacing, and typography are defined here so the entire
dashboard stays visually consistent.
"""

import streamlit as st

# ── Design tokens ─────────────────────────────────────────────────────────────

PALETTE = {
    "bg":          "#0e1117",
    "bg_card":     "#161b22",
    "bg_hover":    "#1c2333",
    "border":      "#21262d",
    "border_hover": "#30363d",
    "text":        "#e6edf3",
    "text_muted":  "#8B949E",
    "text_dim":    "#6e7681",
    "accent":      "#58a6ff",
    "brand":       "#4A90D9",
    "success":     "#27AE60",
    "warning":     "#F39C12",
    "danger":      "#E74C3C",
    "critical":    "#8E44AD",
}

SPACE = {"xs": "4px", "sm": "8px", "md": "16px", "lg": "24px", "xl": "32px", "2xl": "48px"}


# ── Global CSS ────────────────────────────────────────────────────────────────

_GLOBAL_CSS = """
<style>
/* ── Typography ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── Page chrome ────────────────────────────────────────────── */
.main { background: %(bg)s; }
.stSidebar { background: %(bg_card)s; }
h1, h2, h3, h4, p, label, .stMarkdown { color: %(text)s !important; }

/* ── Section dividers ───────────────────────────────────────── */
.section-head {
    display: flex; align-items: center; gap: 10px;
    margin: 32px 0 12px 0;
}
.section-head .icon {
    font-size: 22px; line-height: 1;
}
.section-head .title {
    font-size: 18px; font-weight: 700; color: %(text)s;
    letter-spacing: -0.3px;
}
.section-head .subtitle {
    font-size: 12px; color: %(text_muted)s;
    margin-left: auto; font-weight: 400;
}

/* ── Stat cards (small KPI boxes) ───────────────────────────── */
.stat-card {
    background: %(bg_card)s;
    border: 1px solid %(border)s;
    border-radius: 10px;
    padding: 20px 18px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stat-card:hover {
    border-color: %(border_hover)s;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
}
.stat-card .label {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px;
    color: %(text_muted)s; margin-bottom: 6px; font-weight: 600;
}
.stat-card .value {
    font-size: 28px; font-weight: 800; color: %(text)s;
    letter-spacing: -0.5px; line-height: 1.1;
}
.stat-card .delta {
    font-size: 12px; margin-top: 4px; font-weight: 500;
}
.stat-card .delta.good { color: %(success)s; }
.stat-card .delta.warn { color: %(warning)s; }
.stat-card .delta.bad  { color: %(danger)s; }

/* ── Nav cards (home page) ──────────────────────────────────── */
.nav-card {
    background: %(bg_card)s;
    border: 1px solid %(border)s;
    border-radius: 12px;
    padding: 28px 24px;
    height: 220px;
    box-sizing: border-box;
    display: flex; flex-direction: column;
    transition: border-color 0.25s, box-shadow 0.25s, transform 0.25s;
    cursor: default;
}
.nav-card:hover {
    border-color: %(border_hover)s;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    transform: translateY(-2px);
}
.nav-card .nav-icon { font-size: 28px; margin-bottom: 12px; }
.nav-card .nav-title {
    font-size: 17px; font-weight: 700; color: %(text)s;
    margin-bottom: 8px;
}
.nav-card .nav-desc {
    font-size: 13px; color: %(text_muted)s; line-height: 1.65;
    flex: 1;
}

/* ── Route cards (explorer) ─────────────────────────────────── */
.route-card {
    background: %(bg_card)s; border: 1px solid %(border)s;
    border-radius: 10px; padding: 18px 16px; margin: 4px 0;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.route-card:hover {
    border-color: %(border_hover)s;
    box-shadow: 0 2px 16px rgba(0,0,0,0.3);
}
.route-card h4 { margin: 0 0 10px 0; font-size: 15px; }
.metric-big { font-size: 26px; font-weight: 700; margin: 4px 0; }
.metric-label {
    font-size: 11px; color: %(text_muted)s;
    text-transform: uppercase; letter-spacing: .5px;
}
.tag {
    display: inline-block; padding: 2px 8px;
    border-radius: 12px; font-size: 11px; font-weight: 600;
}

/* ── Wizard animations ──────────────────────────────────────── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0);    }
}
.wiz-wrap { animation: fadeSlideIn 0.3s ease-out; }
@keyframes slideIn {
    from { opacity: 0; transform: translateX(24px); }
    to   { opacity: 1; transform: translateX(0);    }
}
.turnstile-slide { animation: slideIn 0.25s ease-out; }

/* ── Wizard step indicator ──────────────────────────────────── */
.step-indicator {
    display: flex; align-items: center; justify-content: center;
    gap: 0; margin-bottom: 20px;
}
.step-dot {
    width: 32px; height: 32px; border-radius: 50%%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700; transition: all 0.25s;
}
.step-dot.done     { background: %(success)s; color: #fff; }
.step-dot.active   { background: %(accent)s;  color: #fff; box-shadow: 0 0 0 3px %(accent)s44; }
.step-dot.upcoming { background: %(border)s;  color: %(text_muted)s; }
.step-line {
    width: 36px; height: 2px;
}
.step-line.done     { background: %(success)s; }
.step-line.upcoming { background: %(border)s;  }

/* ── Footer ─────────────────────────────────────────────────── */
.sonar-footer {
    margin-top: 64px; padding: 24px 0 12px 0;
    border-top: 1px solid %(border)s;
    text-align: center;
}
.sonar-footer .brand {
    font-size: 14px; font-weight: 700;
    color: %(brand)s; letter-spacing: -0.3px;
}
.sonar-footer .sub {
    font-size: 11px; color: %(text_dim)s;
    margin-top: 4px;
}

/* ── Misc polish ────────────────────────────────────────────── */
.mode-card {
    flex: 1; height: 200px; box-sizing: border-box;
    display: flex; flex-direction: column;
    background: %(bg_card)s; border: 1px solid %(border)s;
    border-radius: 10px; padding: 24px 20px;
    transition: border-color 0.25s, box-shadow 0.25s, transform 0.25s;
}
.mode-card:hover {
    border-color: %(border_hover)s;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    transform: translateY(-2px);
}

/* Streamlit overrides */
div[data-testid="stMetric"] {
    background: %(bg_card)s;
    border: 1px solid %(border)s;
    border-radius: 10px;
    padding: 16px;
}
div[data-testid="stMetric"] label { color: %(text_muted)s !important; font-size: 12px !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 26px !important; font-weight: 700 !important; color: %(text)s !important;
}
</style>
""" % PALETTE


def inject_global_css() -> None:
    """Call once at the top of every page to apply the design system."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


# ── Reusable components ───────────────────────────────────────────────────────

def section_header(icon: str, title: str, subtitle: str = "") -> None:
    """Render a consistent section header with optional right-aligned subtitle."""
    sub_html = f'<span class="subtitle">{subtitle}</span>' if subtitle else ""
    st.markdown(
        f'<div class="section-head">'
        f'<span class="icon">{icon}</span>'
        f'<span class="title">{title}</span>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, delta: str = "", delta_type: str = "good") -> str:
    """Return HTML for a single stat card. Use inside st.markdown(unsafe_allow_html=True)."""
    delta_html = f'<div class="delta {delta_type}">{delta}</div>' if delta else ""
    return (
        f'<div class="stat-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'{delta_html}</div>'
    )


def render_footer() -> None:
    """Render the SONAR branded footer at the bottom of every page."""
    st.markdown(
        '<div class="sonar-footer">'
        '<div class="brand">SONAR</div>'
        '<div class="sub">Supply-chain Optimization &amp; Network Analysis for Resilience'
        ' &middot; UNCTAD Maritime Data 2016\u20132021</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def wizard_step_indicator(current: int, total: int) -> None:
    """Render a numbered step indicator (circles + connecting lines)."""
    parts = []
    for i in range(total):
        cls = "done" if i < current else "active" if i == current else "upcoming"
        icon = "\u2713" if i < current else str(i + 1)
        parts.append(f'<div class="step-dot {cls}">{icon}</div>')
        if i < total - 1:
            line_cls = "done" if i < current else "upcoming"
            parts.append(f'<div class="step-line {line_cls}"></div>')
    st.markdown(
        f'<div class="step-indicator">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )
