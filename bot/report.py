"""Rapport du soir, en HTML autonome (aucune dependance, aucun CDN).

Le rapport repond a quatre questions, dans cet ordre :
  1. Combien vaut le portefeuille, et par rapport a quoi ?
  2. Qu'est-ce que le bot a fait aujourd'hui ?
  3. Qu'est-ce qu'il a decide de NE PAS faire, et pourquoi ?
  4. Est-ce que ca tient sur la duree (drawdown, score, comparaison marche) ?

La troisieme est celle que les rapports de trading oublient toujours,
et c'est souvent la plus instructive.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .config import Config
from .engine import RunResult
from .forward import ForwardLog
from .models import Action, Candle

# --- Palette (validee : contraste et distinction daltonisme en clair et sombre)
SERIES_BOT_LIGHT = "#2a78d6"
SERIES_BH_LIGHT = "#eb6834"
SERIES_BOT_DARK = "#3987e5"
SERIES_BH_DARK = "#d95926"

MAX_POINTS = 420  # au-dela, la courbe est sous-echantillonnee


# ----------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------
def quote_currency(symbol: str) -> str:
    for suffix in ("USDT", "BUSD", "USDC", "USD", "EUR"):
        if symbol.upper().endswith(suffix):
            return suffix
    return ""


def fmt_num(value: float, decimals: int = 2) -> str:
    """Format francais : espace fine pour les milliers, virgule decimale."""
    s = f"{value:,.{decimals}f}"
    return s.replace(",", " ").replace(".", ",")


def fmt_pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:+.{decimals}f}".replace(".", ",") + " %"


def fmt_signed(value: float, decimals: int = 2) -> str:
    sign = "+" if value >= 0 else ""
    return sign + fmt_num(value, decimals)


def local_dt(ts: int, offset_hours: float) -> datetime:
    return datetime.fromtimestamp(ts / 1000, tz=timezone(timedelta(hours=offset_hours)))


def fmt_time(ts: int, offset_hours: float) -> str:
    return local_dt(ts, offset_hours).strftime("%d/%m %H:%M")


def day_bounds(day: str, offset_hours: float) -> Tuple[int, int]:
    """Bornes en millisecondes d'une journee locale 'AAAA-MM-JJ'."""
    tz = timezone(timedelta(hours=offset_hours))
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=tz)
    end = start + timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000) - 1


# ----------------------------------------------------------------------
# Courbe d'equite en SVG
# ----------------------------------------------------------------------
@dataclass
class Series:
    label: str
    points: List[Tuple[int, float]]
    color_light: str
    color_dark: str
    key: str
    short: str = ""   # etiquette courte affichee en bout de ligne

    def __post_init__(self) -> None:
        if not self.short:
            self.short = self.label


def _downsample(points: Sequence[Tuple[int, float]], limit: int) -> List[Tuple[int, float]]:
    if len(points) <= limit:
        return list(points)
    step = len(points) / limit
    out = [points[int(i * step)] for i in range(limit)]
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


def equity_chart(
    series: Sequence[Series],
    offset_hours: float,
    quote: str,
    clamp_min_zero: bool = True,
    aria_label: str = "Courbe d'equite du bot comparee a acheter et conserver",
) -> str:
    """Courbe generique a un seul axe Y (meme unite pour toutes les series),
    avec legende, etiquettes directes en bout de ligne et survol.

    `clamp_min_zero` : vrai pour une courbe de capital (qui ne descend
    jamais sous zero) ; faux pour un P&L cumule, qui doit pouvoir montrer
    une serie de pertes sans etre coupee au ras de l'axe.
    """
    if not series or not series[0].points:
        return '<p class="empty">Pas encore assez de donnees pour tracer la courbe.</p>'

    W, H = 920, 340
    pad_l, pad_r, pad_t, pad_b = 62, 96, 20, 34

    all_pts = [p for s in series for p in s.points]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_max == y_min:
        y_max = y_min + 1
    margin = (y_max - y_min) * 0.08
    y_min = y_min - margin
    if clamp_min_zero:
        # Le capital ne peut pas etre negatif : on ne descend jamais l'axe sous zero.
        y_min = max(0.0, y_min)
    y_max = y_max + margin
    x_span = max(x_max - x_min, 1)

    def sx(ts: int) -> float:
        return pad_l + (ts - x_min) / x_span * (W - pad_l - pad_r)

    def sy(v: float) -> float:
        return H - pad_b - (v - y_min) / (y_max - y_min) * (H - pad_t - pad_b)

    parts: List[str] = []
    parts.append(
        f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="{html.escape(aria_label)}" '
        f'preserveAspectRatio="xMidYMid meet">'
    )

    # Grille horizontale + graduations (5 niveaux)
    for k in range(5):
        v = y_min + (y_max - y_min) * k / 4
        y = sy(v)
        parts.append(
            f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{W - pad_r}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{pad_l - 10}" y="{y + 4:.1f}" text-anchor="end">'
            f"{fmt_num(v, 0)}</text>"
        )

    # Graduations de temps (4 reperes)
    for k in range(4):
        ts = int(x_min + x_span * k / 3)
        x = sx(ts)
        anchor = "start" if k == 0 else ("end" if k == 3 else "middle")
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{H - pad_b + 20:.1f}" text-anchor="{anchor}">'
            f"{local_dt(ts, offset_hours).strftime('%d/%m/%y')}</text>"
        )

    parts.append(
        f'<line class="axis" x1="{pad_l}" y1="{H - pad_b}" x2="{W - pad_r}" y2="{H - pad_b}"/>'
    )

    # Lignes
    for s in series:
        pts = _downsample(s.points, MAX_POINTS)
        d = " ".join(
            ("M" if k == 0 else "L") + f"{sx(ts):.1f},{sy(v):.1f}" for k, (ts, v) in enumerate(pts)
        )
        parts.append(f'<path class="line line-{s.key}" d="{d}"/>')
        # Etiquette directe en bout de ligne : l'identite ne repose jamais
        # sur la couleur seule.
        last_ts, last_v = pts[-1]
        parts.append(
            f'<text class="endlabel end-{s.key}" x="{sx(last_ts) + 8:.1f}" '
            f'y="{sy(last_v) + 4:.1f}">{html.escape(s.short)}</text>'
        )

    # Couche de survol
    parts.append(f'<line class="crosshair" id="crosshair" x1="0" y1="{pad_t}" x2="0" y2="{H - pad_b}" style="opacity:0"/>')
    for s in series:
        parts.append(f'<circle class="dot dot-{s.key}" id="dot-{s.key}" r="4.5" style="opacity:0"/>')
    parts.append(
        f'<rect id="hitarea" x="{pad_l}" y="{pad_t}" width="{W - pad_l - pad_r}" '
        f'height="{H - pad_t - pad_b}" fill="transparent"/>'
    )
    parts.append("</svg>")

    # Donnees pour le survol (serialisees simplement, sans dependance)
    data_js = []
    for s in series:
        pts = _downsample(s.points, MAX_POINTS)
        data_js.append(
            "{key:%r,label:%r,pts:[%s]}"
            % (
                s.key,
                s.label,
                ",".join(f"[{ts},{v:.4f}]" for ts, v in pts),
            )
        )

    script = f"""
<script>
(function(){{
  var W={W},H={H},padL={pad_l},padR={pad_r},padT={pad_t},padB={pad_b};
  var xMin={x_min},xSpan={x_span},yMin={y_min:.6f},yMax={y_max:.6f};
  var series=[{','.join(data_js)}];
  var svg=document.querySelector('.chart');
  var hit=document.getElementById('hitarea');
  var cross=document.getElementById('crosshair');
  var tip=document.getElementById('tooltip');
  if(!svg||!hit||!tip) return;
  function sx(ts){{return padL+(ts-xMin)/xSpan*(W-padL-padR);}}
  function sy(v){{return H-padB-(v-yMin)/(yMax-yMin)*(H-padT-padB);}}
  function fmt(v){{return v.toLocaleString('fr-FR',{{minimumFractionDigits:2,maximumFractionDigits:2}});}}
  function move(evt){{
    var rect=svg.getBoundingClientRect();
    var px=(evt.clientX-rect.left)/rect.width*W;
    var ts=xMin+(px-padL)/(W-padL-padR)*xSpan;
    var rows='',cx=null,stamp='';
    series.forEach(function(s){{
      var best=s.pts[0],bd=Infinity;
      for(var i=0;i<s.pts.length;i++){{
        var d=Math.abs(s.pts[i][0]-ts);
        if(d<bd){{bd=d;best=s.pts[i];}}
      }}
      cx=sx(best[0]);
      if(!stamp){{
        stamp=new Date(best[0]).toLocaleString('fr-FR',{{day:'2-digit',month:'2-digit',year:'2-digit',hour:'2-digit',minute:'2-digit'}});
      }}
      var dot=document.getElementById('dot-'+s.key);
      if(dot){{dot.setAttribute('cx',cx);dot.setAttribute('cy',sy(best[1]));dot.style.opacity=1;}}
      rows+='<div class="tiprow"><span class="swatch sw-'+s.key+'"></span>'+
            '<span class="tiplabel">'+s.label+'</span>'+
            '<span class="tipval">'+fmt(best[1])+' {html.escape(quote)}</span></div>';
    }});
    cross.setAttribute('x1',cx);cross.setAttribute('x2',cx);cross.style.opacity=1;
    tip.innerHTML='<div class="tipdate">'+stamp+'</div>'+rows;
    tip.style.opacity=1;
    var host=svg.parentElement.getBoundingClientRect();
    var left=evt.clientX-host.left+14;
    if(left+190>host.width) left=evt.clientX-host.left-200;
    tip.style.left=left+'px';
    tip.style.top=(evt.clientY-host.top-10)+'px';
  }}
  function leave(){{
    cross.style.opacity=0;tip.style.opacity=0;
    series.forEach(function(s){{var d=document.getElementById('dot-'+s.key);if(d)d.style.opacity=0;}});
  }}
  hit.addEventListener('mousemove',move);
  hit.addEventListener('mouseleave',leave);
}})();
</script>
"""
    legend = '<div class="legend">' + "".join(
        f'<span class="legend-item"><span class="swatch sw-{s.key}"></span>{html.escape(s.label)}</span>'
        for s in series
    ) + "</div>"

    return (
        legend
        + '<div class="chartwrap">'
        + "".join(parts)
        + '<div class="tooltip" id="tooltip"></div>'
        + "</div>"
        + script
    )


# ----------------------------------------------------------------------
# Rendu
# ----------------------------------------------------------------------
def _stat(label: str, value: str, tone: str = "", note: str = "") -> str:
    note_html = f'<div class="statnote">{html.escape(note)}</div>' if note else ""
    return (
        f'<div class="stat"><div class="statlabel">{html.escape(label)}</div>'
        f'<div class="statvalue {tone}">{value}</div>{note_html}</div>'
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], empty: str) -> str:
    if not rows:
        return f'<p class="empty">{html.escape(empty)}</p>'
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def build_report(
    cfg: Config,
    result: RunResult,
    candles: Sequence[Candle],
    day: Optional[str] = None,
    mode_label: str = "backtest",
    forward_stats: Optional[dict] = None,
) -> str:
    off = cfg.report.timezone_offset_hours
    quote = quote_currency(cfg.market.symbol)
    m = result.metrics
    assert m is not None

    # --- series -----------------------------------------------------
    bot_series = Series(
        "Bot", list(result.equity_curve), SERIES_BOT_LIGHT, SERIES_BOT_DARK, "bot", "Bot"
    )
    used = list(candles[result.warmup_bars + 1:])
    bh_points: List[Tuple[int, float]] = []
    if used:
        base = used[0].close
        for c in used:
            bh_points.append((c.ts, cfg.portfolio.initial_cash * c.close / base))
    bh_series = Series(
        "Acheter et conserver", bh_points, SERIES_BH_LIGHT, SERIES_BH_DARK, "bh", "Marche"
    )

    # --- perimetre du jour ------------------------------------------
    if day:
        start_ms, end_ms = day_bounds(day, off)
        title_period = f"Journee du {datetime.strptime(day, '%Y-%m-%d').strftime('%d/%m/%Y')}"
    else:
        start_ms = result.equity_curve[0][0] if result.equity_curve else 0
        end_ms = result.equity_curve[-1][0] if result.equity_curve else 0
        title_period = "Periode complete"

    day_trades = [t for t in result.trades if start_ms <= t.exit_ts <= end_ms]
    day_decisions = [d for d in result.decisions if start_ms <= d.ts <= end_ms]
    day_abstentions = [d for d in day_decisions if d.action == Action.ABSTAIN]
    day_scores = [e for e in result.score_entries if start_ms <= e.ts <= end_ms]
    day_score = round(sum(e.value for e in day_scores), 3)
    day_pnl = round(sum(t.pnl for t in day_trades), 2)

    eq_in_day = [(ts, v) for ts, v in result.equity_curve if start_ms <= ts <= end_ms]
    day_equity_change = 0.0
    if eq_in_day:
        prior = [v for ts, v in result.equity_curve if ts < eq_in_day[0][0]]
        opening = prior[-1] if prior else cfg.portfolio.initial_cash
        day_equity_change = (eq_in_day[-1][1] - opening) / opening if opening else 0.0

    tone = lambda v: "good" if v > 0 else ("bad" if v < 0 else "")  # noqa: E731

    # --- blocs -------------------------------------------------------
    suffix = "du jour" if day else "de la periode"
    stats = "".join([
        _stat("Capital", f"{fmt_num(m.final_equity)} {quote}",
              note=f"depart {fmt_num(m.initial_equity)} {quote}"),
        _stat(f"Variation {suffix}", fmt_pct(day_equity_change), tone(day_equity_change),
              note=f"resultat realise {fmt_signed(day_pnl)} {quote}"),
        _stat(f"Score {suffix}", fmt_signed(day_score, 2), tone(day_score),
              note=f"cumule {fmt_signed(result.score_breakdown.get('total', 0.0), 2)}"),
        _stat(f"Trades {suffix}", str(len(day_trades)),
              note=f"{len(day_abstentions)} abstention(s)"),
        _stat("Drawdown max", fmt_pct(-m.max_drawdown), "bad" if m.max_drawdown > 0 else "",
              note="pire perte cumulee traversee"),
        _stat("vs marche", fmt_pct(m.excess_return), tone(m.excess_return),
              note=f"bot {fmt_pct(m.total_return)} / marche {fmt_pct(m.buy_hold_return)}"),
    ])

    if forward_stats:
        jours = forward_stats.get("jours_de_forward", 0)
        note = f"{forward_stats.get('decisions_live', 0)} decisions en direct"
        if forward_stats.get("divergences"):
            note += f" - {forward_stats['divergences']} divergence(s)"
        elif jours < 90:
            note += " - moins de 3 mois, trop tot pour conclure"
        stats += _stat(
            "Forward testing",
            f"{fmt_num(jours, 0)} j",
            "bad" if forward_stats.get("divergences") else "",
            note=note,
        )

    MAX_ROWS = 60
    shown_trades = day_trades[-MAX_ROWS:]
    trade_note = ""
    if len(day_trades) > MAX_ROWS:
        trade_note = (
            f'<p class="note">({len(day_trades) - MAX_ROWS} trades plus anciens non affiches ; '
            "l'historique complet est en base.)</p>"
        )
    trade_rows = [
        [
            fmt_time(t.entry_ts, off),
            fmt_time(t.exit_ts, off),
            fmt_num(t.entry_price),
            fmt_num(t.exit_price),
            f'<span class="{tone(t.pnl)}">{fmt_signed(t.pnl)}</span>',
            f'<span class="{tone(t.pnl)}">{fmt_pct(t.pnl_pct)}</span>',
            f'<span class="{tone(t.r_multiple)}">{fmt_signed(t.r_multiple)} R</span>',
            html.escape(t.exit_reason),
        ]
        for t in shown_trades
    ]

    abst_rows = [
        [fmt_time(d.ts, off), fmt_num(d.price), html.escape(d.reason)]
        for d in day_abstentions[-MAX_ROWS:]
    ]
    abst_note = ""
    if len(day_abstentions) > MAX_ROWS:
        abst_note = (
            f'<p class="note">({len(day_abstentions) - MAX_ROWS} abstentions plus anciennes '
            "non affichees ; l'historique complet est en base.)</p>"
        )

    bd = result.score_breakdown
    pf_txt = "infini" if m.profit_factor == float("inf") else fmt_num(m.profit_factor)
    metric_rows = [
        ["Rendement total", fmt_pct(m.total_return)],
        ["Rendement acheter-et-conserver", fmt_pct(m.buy_hold_return)],
        ["Ecart avec le marche", fmt_pct(m.excess_return)],
        ["Drawdown maximum", fmt_pct(-m.max_drawdown) + f" (du {m.max_drawdown_start} au {m.max_drawdown_end})"],
        ["Plus longue periode sous le sommet", f"{m.longest_drawdown_bars} bougies"],
        ["Rendement annualise", fmt_pct(m.annualized_return)],
        ["Volatilite annualisee", fmt_pct(m.annualized_volatility)],
        ["Ratio de Sharpe", fmt_num(m.sharpe)],
        ["Nombre de trades", str(m.trade_count)],
        ["Taux de reussite", fmt_pct(m.win_rate).replace("+", "")],
        ["Gain moyen / perte moyenne", f"{fmt_num(m.avg_win)} / {fmt_num(m.avg_loss)}"],
        ["Facteur de profit", pf_txt],
        ["Esperance par trade", f"{fmt_signed(m.expectancy_r)} R"],
        ["Meilleur / pire trade", f"{fmt_signed(m.best_trade_r)} R / {fmt_signed(m.worst_trade_r)} R"],
        ["Frais cumules", f"{fmt_num(m.total_fees)} {quote}"],
        ["Exposition au marche", fmt_pct(m.exposure).replace("+", "")],
        ["Jours sans aucun trade", f"{m.days_without_trade} sur {int(m.days_covered)} jours"],
        ["Score cumule", fmt_signed(bd.get("total", 0.0))],
        ["  dont trades", fmt_signed(bd.get("trade_score", 0.0))],
        ["  dont abstentions", fmt_signed(bd.get("abstention_score", 0.0))],
        ["Score moyen par decision notee", fmt_signed(bd.get("moyenne_par_note", 0.0))],
        ["Abstentions penalisees / recompensees",
         f"{bd.get('abstentions_penalisees', 0)} / {bd.get('abstentions_recompensees', 0)}"],
    ]

    halted = ""
    if result.halted_days:
        recent = ", ".join(result.halted_days[-5:])
        halted = (
            f'<div class="alert"><strong>Coupe-circuit journalier declenche</strong> sur '
            f"{len(result.halted_days)} journee(s). Dernieres : {html.escape(recent)}.</div>"
        )

    generated = datetime.now(timezone(timedelta(hours=off))).strftime("%d/%m/%Y a %H:%M")

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rapport {html.escape(cfg.market.symbol)} - {html.escape(title_period)}</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1: #fcfcfb; --page: #f9f9f7;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,.10);
    --series-bot: {SERIES_BOT_LIGHT}; --series-bh: {SERIES_BH_LIGHT};
    --good: #006300; --bad: #d03b3b; --warn-bg: #fff6e5; --warn-border: #fab219;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --surface-1: #1a1a19; --page: #0d0d0d;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
      --series-bot: {SERIES_BOT_DARK}; --series-bh: {SERIES_BH_DARK};
      --good: #0ca30c; --bad: #e66767; --warn-bg: #2a2313; --warn-border: #fab219;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
    --series-bot: {SERIES_BOT_DARK}; --series-bh: {SERIES_BH_DARK};
    --good: #0ca30c; --bad: #e66767; --warn-bg: #2a2313; --warn-border: #fab219;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--page); color: var(--text-primary);
    font: 14px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 1000px; margin: 0 auto; padding: 28px 20px 64px; }}
  header {{ margin-bottom: 22px; }}
  h1 {{ font-size: 21px; margin: 0 0 4px; letter-spacing: -.01em; }}
  .sub {{ color: var(--text-secondary); font-size: 13px; }}
  .card {{
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px 20px; margin-bottom: 18px;
  }}
  h2 {{ font-size: 14px; margin: 0 0 14px; text-transform: uppercase;
       letter-spacing: .06em; color: var(--text-secondary); font-weight: 600; }}
  .stats {{ display: grid; background: var(--surface-1);
            grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
            border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
            margin-bottom: 18px; }}
  .stat {{ padding: 14px 16px; border-right: 1px solid var(--border);
           border-bottom: 1px solid var(--border); }}
  .statlabel {{ font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
                color: var(--muted); margin-bottom: 5px; }}
  .statvalue {{ font-size: 20px; font-weight: 600; letter-spacing: -.02em; }}
  .statnote {{ font-size: 11.5px; color: var(--text-secondary); margin-top: 3px; }}
  .good {{ color: var(--good); }}
  .bad {{ color: var(--bad); }}
  .chartwrap {{ position: relative; }}
  .chart {{ width: 100%; height: auto; display: block; overflow: visible; }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }}
  .axis {{ stroke: var(--axis); stroke-width: 1; }}
  .tick {{ fill: var(--muted); font-size: 10.5px; font-variant-numeric: tabular-nums; }}
  .line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
  .line-bot {{ stroke: var(--series-bot); }}
  .line-bh {{ stroke: var(--series-bh); stroke-dasharray: 5 4; }}
  .endlabel {{ font-size: 11px; font-weight: 600; fill: var(--text-secondary); }}
  .crosshair {{ stroke: var(--axis); stroke-width: 1; stroke-dasharray: 3 3; }}
  .dot {{ stroke: var(--surface-1); stroke-width: 2; }}
  .dot-bot {{ fill: var(--series-bot); }}
  .dot-bh {{ fill: var(--series-bh); }}
  .legend {{ display: flex; gap: 18px; margin-bottom: 8px; font-size: 12.5px;
             color: var(--text-secondary); }}
  .legend-item {{ display: inline-flex; align-items: center; gap: 7px; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; }}
  .sw-bot {{ background: var(--series-bot); }}
  .sw-bh {{ background: var(--series-bh); }}
  .tooltip {{
    position: absolute; pointer-events: none; opacity: 0; transition: opacity .1s;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
    padding: 8px 10px; font-size: 12px; min-width: 180px;
    box-shadow: 0 4px 14px rgba(0,0,0,.12);
  }}
  .tipdate {{ color: var(--muted); font-size: 11px; margin-bottom: 5px; }}
  .tiprow {{ display: flex; align-items: center; gap: 7px; }}
  .tiplabel {{ color: var(--text-secondary); }}
  .tipval {{ margin-left: auto; font-variant-numeric: tabular-nums; font-weight: 600; }}
  .tablewrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th {{ text-align: left; font-weight: 600; color: var(--text-secondary);
        font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
        padding: 6px 10px 6px 0; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  td {{ padding: 7px 10px 7px 0; border-bottom: 1px solid var(--border);
        vertical-align: top; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  td:last-child {{ font-variant-numeric: normal; color: var(--text-secondary);
                   white-space: normal; }}
  .empty {{ color: var(--muted); font-style: italic; margin: 0; }}
  .note {{ color: var(--muted); font-size: 12px; }}
  .alert {{ background: var(--warn-bg); border-left: 3px solid var(--warn-border);
            padding: 10px 14px; border-radius: 6px; margin-bottom: 18px; font-size: 13px; }}
  footer {{ color: var(--muted); font-size: 12px; margin-top: 26px;
            border-top: 1px solid var(--border); padding-top: 14px; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>{html.escape(cfg.market.symbol)} - {html.escape(title_period)}</h1>
    <div class="sub">
      Portefeuille fictif - aucun argent reel engage - {html.escape(mode_label)} -
      unite de temps {html.escape(cfg.market.timeframe)} - genere le {generated}
    </div>
  </header>

  {halted}

  <div class="stats">{stats}</div>

  <div class="card">
    <h2>Courbe d'equite</h2>
    {equity_chart([bot_series, bh_series], off, quote)}
  </div>

  <div class="card">
    <h2>Trades cloture(s) sur la periode</h2>
    {_table(
        ["Entree", "Sortie", "Prix entree", "Prix sortie", "Resultat", "%", "R", "Motif de sortie"],
        trade_rows,
        "Aucun trade cloture sur cette periode.",
    )}
    {trade_note}
  </div>

  <div class="card">
    <h2>Decisions de ne pas trader</h2>
    {_table(["Heure", "Prix", "Motif"], abst_rows, "Aucune abstention enregistree sur cette periode.")}
    {abst_note}
  </div>

  <div class="card">
    <h2>Metriques cumulees</h2>
    {_table(["Metrique", "Valeur"], [[html.escape(a), b] for a, b in metric_rows], "")}
  </div>

  <footer>
    Rapport genere automatiquement. Les resultats portent sur un portefeuille
    simule : ils ne tiennent pas compte de la liquidite reelle du carnet d'ordres
    ni de la latence d'execution, et ne constituent ni une preuve de rentabilite
    ni un conseil en investissement.
  </footer>
</div>
</body>
</html>
"""


def build_forward_report(
    cfg: Config,
    log: ForwardLog,
    since_ts: Optional[int] = None,
    since_label: Optional[str] = None,
) -> str:
    """Page HTML autonome : ce que le forward test EN DIRECT a reellement
    gagne ou perdu, par opposition a la 'Courbe d'equite' du rapport du
    soir qui rejoue tout l'historique et se termine toujours a aujourd'hui.

    Construite uniquement a partir du journal append-seul de forward.py :
    aucune valeur ici n'a pu etre calculee avec la connaissance du futur.
    """
    off = cfg.report.timezone_offset_hours
    quote = quote_currency(cfg.market.symbol)
    s = log.stats(since_ts=since_ts)
    trades = log.live_trades(since_ts=since_ts)

    # --- courbe de P&L cumule : part toujours de 0, pas d'un capital ----
    points: List[Tuple[int, float]] = []
    running = 0.0
    for t in trades:
        running += float(t.get("pnl", 0.0))
        points.append((int(t["exit_ts"]), round(running, 4)))
    pnl_series = Series(
        "P&L cumule", points, SERIES_BOT_LIGHT, SERIES_BOT_DARK, "bot", "P&L"
    )

    tone = lambda v: "good" if v > 0 else ("bad" if v < 0 else "")  # noqa: E731

    perimetre = f"depuis {since_label}" if since_label else "depuis le debut du journal"
    stats_html = "".join([
        _stat("P&L realise", f"{fmt_signed(s['pnl_total'])} {quote}", tone(s["pnl_total"]),
              note=perimetre),
        _stat("Trades clotures", str(s["trades_live"]),
              note=f"{s['trades_gagnants']} gagnant(s) / {s['trades_perdants']} perdant(s)"),
        _stat("Decisions en direct", str(s["decisions_live"]),
              note="jamais recalculees apres coup"),
        _stat("Duree de la fenetre", f"{fmt_num(s['jours_de_forward'], 1)} j",
              note=f"depuis {s['debut_live'] or '-'}"),
    ])
    if s["divergences"]:
        stats_html += _stat(
            "Divergences", str(s["divergences"]), "bad",
            note="forward test rompu sur cette fenetre",
        )

    trade_rows = [
        [
            fmt_time(int(t["entry_ts"]), off),
            fmt_time(int(t["exit_ts"]), off),
            fmt_num(float(t["entry_price"])),
            fmt_num(float(t["exit_price"])),
            f'<span class="{tone(float(t["pnl"]))}">{fmt_signed(float(t["pnl"]))}</span>',
            f'<span class="{tone(float(t["r_multiple"]))}">{fmt_signed(float(t["r_multiple"]))} R</span>',
            html.escape(str(t.get("exit_reason", ""))),
        ]
        for t in reversed(trades)
    ]

    generated = datetime.now(timezone(timedelta(hours=off))).strftime("%d/%m/%Y a %H:%M")
    titre_fenetre = f" - {html.escape(since_label)}" if since_label else ""

    warning = ""
    if not trades:
        warning = (
            '<div class="alert">Aucun trade clôture pour l\'instant sur cette fenetre. '
            "C'est attendu tant que le forward test vient de commencer : la courbe "
            "apparaitra au fur et a mesure que des trades se cloturent reellement.</div>"
        )

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forward test reel {html.escape(cfg.market.symbol)}{titre_fenetre}</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1: #fcfcfb; --page: #f9f9f7;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,.10);
    --series-bot: {SERIES_BOT_LIGHT};
    --good: #006300; --bad: #d03b3b; --warn-bg: #fff6e5; --warn-border: #fab219;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --surface-1: #1a1a19; --page: #0d0d0d;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
      --series-bot: {SERIES_BOT_DARK};
      --good: #0ca30c; --bad: #e66767; --warn-bg: #2a2313; --warn-border: #fab219;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
    --series-bot: {SERIES_BOT_DARK};
    --good: #0ca30c; --bad: #e66767; --warn-bg: #2a2313; --warn-border: #fab219;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--page); color: var(--text-primary);
    font: 14px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 1000px; margin: 0 auto; padding: 28px 20px 64px; }}
  header {{ margin-bottom: 22px; }}
  h1 {{ font-size: 21px; margin: 0 0 4px; letter-spacing: -.01em; }}
  .sub {{ color: var(--text-secondary); font-size: 13px; }}
  .card {{
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px 20px; margin-bottom: 18px;
  }}
  h2 {{ font-size: 14px; margin: 0 0 14px; text-transform: uppercase;
       letter-spacing: .06em; color: var(--text-secondary); font-weight: 600; }}
  .stats {{ display: grid; background: var(--surface-1);
            grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
            border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
            margin-bottom: 18px; }}
  .stat {{ padding: 14px 16px; border-right: 1px solid var(--border);
           border-bottom: 1px solid var(--border); }}
  .statlabel {{ font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
                color: var(--muted); margin-bottom: 5px; }}
  .statvalue {{ font-size: 20px; font-weight: 600; letter-spacing: -.02em; }}
  .statnote {{ font-size: 11.5px; color: var(--text-secondary); margin-top: 3px; }}
  .good {{ color: var(--good); }}
  .bad {{ color: var(--bad); }}
  .chartwrap {{ position: relative; }}
  .chart {{ width: 100%; height: auto; display: block; overflow: visible; }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }}
  .axis {{ stroke: var(--axis); stroke-width: 1; }}
  .tick {{ fill: var(--muted); font-size: 10.5px; font-variant-numeric: tabular-nums; }}
  .line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
  .line-bot {{ stroke: var(--series-bot); }}
  .endlabel {{ font-size: 11px; font-weight: 600; fill: var(--text-secondary); }}
  .crosshair {{ stroke: var(--axis); stroke-width: 1; stroke-dasharray: 3 3; }}
  .dot {{ stroke: var(--surface-1); stroke-width: 2; }}
  .dot-bot {{ fill: var(--series-bot); }}
  .legend {{ display: flex; gap: 18px; margin-bottom: 8px; font-size: 12.5px;
             color: var(--text-secondary); }}
  .legend-item {{ display: inline-flex; align-items: center; gap: 7px; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; }}
  .sw-bot {{ background: var(--series-bot); }}
  .tooltip {{
    position: absolute; pointer-events: none; opacity: 0; transition: opacity .1s;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
    padding: 8px 10px; font-size: 12px; min-width: 180px;
    box-shadow: 0 4px 14px rgba(0,0,0,.12);
  }}
  .tipdate {{ color: var(--muted); font-size: 11px; margin-bottom: 5px; }}
  .tiprow {{ display: flex; align-items: center; gap: 7px; }}
  .tiplabel {{ color: var(--text-secondary); }}
  .tipval {{ margin-left: auto; font-variant-numeric: tabular-nums; font-weight: 600; }}
  .tablewrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th {{ text-align: left; font-weight: 600; color: var(--text-secondary);
        font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
        padding: 6px 10px 6px 0; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  td {{ padding: 7px 10px 7px 0; border-bottom: 1px solid var(--border);
        vertical-align: top; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  td:last-child {{ font-variant-numeric: normal; color: var(--text-secondary);
                   white-space: normal; }}
  .empty {{ color: var(--muted); font-style: italic; margin: 0; }}
  .note {{ color: var(--muted); font-size: 12px; }}
  .alert {{ background: var(--warn-bg); border-left: 3px solid var(--warn-border);
            padding: 10px 14px; border-radius: 6px; margin-bottom: 18px; font-size: 13px; }}
  footer {{ color: var(--muted); font-size: 12px; margin-top: 26px;
            border-top: 1px solid var(--border); padding-top: 14px; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>{html.escape(cfg.market.symbol)} - forward test reel{titre_fenetre}</h1>
    <div class="sub">
      Portefeuille fictif - construit uniquement a partir du journal en direct
      (jamais du backtest rejoue) - genere le {generated}
    </div>
  </header>

  {warning}

  <div class="stats">{stats_html}</div>

  <div class="card">
    <h2>P&amp;L cumule (trades reellement clotures en direct)</h2>
    {equity_chart(
        [pnl_series], off, quote, clamp_min_zero=False,
        aria_label="P&L cumule reellement realise en forward testing",
    )}
  </div>

  <div class="card">
    <h2>Trades du journal en direct</h2>
    {_table(
        ["Entree", "Sortie", "Prix entree", "Prix sortie", "Resultat", "R", "Motif de sortie"],
        trade_rows,
        "Aucun trade cloture pour l'instant.",
    )}
  </div>

  <footer>
    Ce rapport ne reflete que ce qui est ecrit dans le journal append-seul
    de forward testing : aucune ligne n'a pu etre calculee avec la
    connaissance du futur. Portefeuille simule, aucun argent reel engage,
    aucun ordre reel passe.
  </footer>
</div>
</body>
</html>
"""


def write_report(cfg: Config, html_text: str, filename: str) -> Path:
    out_dir = cfg.path(cfg.report.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(html_text, encoding="utf-8")
    return path
