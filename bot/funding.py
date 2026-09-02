"""Taux de financement des contrats perpetuels Binance.

Pourquoi cette donnee et pas une autre
---------------------------------------
Tous les signaux testes jusqu'ici derivent du PRIX. Ils sont dans tous
les logiciels de graphique de la planete, calcules par des milliers de
gens en permanence. Qu'ils ne contiennent plus d'avantage exploitable
est l'issue attendue.

Le taux de financement est different. C'est le paiement periodique
(toutes les 8 heures) entre acheteurs et vendeurs de contrats perpetuels,
qui maintient le prix du perpetuel colle au prix comptant. Il ne mesure
pas le prix : il mesure le DESEQUILIBRE DE POSITIONNEMENT.

  taux tres positif -> les acheteurs paient cher pour rester acheteurs,
                       donc ils sont nombreux et effet de levier eleve
  taux tres negatif -> ce sont les vendeurs qui paient : positionnement
                       vendeur sature

L'hypothese, documentee dans la litterature sur les marches a terme, est
qu'un positionnement sature se retourne : quand tout le monde est du meme
cote, il ne reste personne pour pousser dans ce sens, et les liquidations
amplifient le mouvement inverse.

C'est une donnee publique et gratuite, mais absente des logiciels de
graphique grand public. Si un avantage accessible existe encore quelque
part sur BTC, c'est plutot de ce cote-la que du cote d'une enieme moyenne
mobile.

Rien ne garantit que ca marche. C'est justement ce qu'on va mesurer.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .models import Candle

FUTURES_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
MAX_LIMIT = 1000
INTERVAL_MS = 8 * 3600 * 1000  # un paiement toutes les 8 heures


class FundingError(RuntimeError):
    pass


def _cache_path(cache_dir: str | Path, symbol: str) -> Path:
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"funding_{symbol}.csv"


def _read_cache(path: Path) -> List[Tuple[int, float]]:
    if not path.exists():
        return []
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append((int(row["ts"]), float(row["rate"])))
    return out


def _write_cache(path: Path, rows: Sequence[Tuple[int, float]]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ts", "rate"])
        w.writerows(rows)
    tmp.replace(path)


def fetch(symbol: str, start_ms: int, end_ms: int, cache_dir: str = "data") -> List[Tuple[int, float]]:
    """Recupere l'historique des taux de financement, avec cache local.

    Endpoint public, sans cle API, en lecture seule — comme le reste du
    projet, ce module est incapable de passer le moindre ordre.
    """
    path = _cache_path(cache_dir, symbol)
    cached = _read_cache(path)
    if cached and cached[0][0] <= start_ms and cached[-1][0] >= end_ms - INTERVAL_MS:
        return [r for r in cached if start_ms <= r[0] <= end_ms]

    fresh: List[Tuple[int, float]] = []
    cursor = start_ms
    while cursor < end_ms:
        url = (
            f"{FUTURES_URL}?symbol={symbol}&startTime={cursor}"
            f"&endTime={end_ms}&limit={MAX_LIMIT}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "paper-trading-bot/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                rows = json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 451:
                raise FundingError(
                    "Binance Futures refuse l'acces depuis ce pays (451). "
                    "Les signaux de financement ne seront pas testables ici."
                ) from exc
            raise FundingError(f"Binance Futures a repondu HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise FundingError(f"Impossible de joindre Binance Futures ({exc.reason})") from exc

        if not rows:
            break
        for r in rows:
            fresh.append((int(r["fundingTime"]), float(r["fundingRate"])))
        last = int(rows[-1]["fundingTime"])
        if last + INTERVAL_MS <= cursor:
            break
        cursor = last + INTERVAL_MS
        if len(rows) < MAX_LIMIT:
            break
        time.sleep(0.2)

    merged = {ts: rate for ts, rate in cached}
    merged.update({ts: rate for ts, rate in fresh})
    allrows = [(ts, merged[ts]) for ts in sorted(merged)]
    if allrows:
        _write_cache(path, allrows)
    return [r for r in allrows if start_ms <= r[0] <= end_ms]


def percentile_events(
    funding: Sequence[Tuple[int, float]],
    window: int = 90,       # 90 paiements = 30 jours d'historique glissant
    low_pct: float = 0.10,
    high_pct: float = 0.90,
) -> Tuple[List[int], List[int]]:
    """Horodatages ou le taux franchit ses extremes recents.

    Le rang est calcule sur une fenetre GLISSANTE ANTERIEURE uniquement :
    a chaque instant, on ne compare le taux qu'aux `window` paiements qui
    l'ont precede. Utiliser les percentiles calcules sur toute la periode
    ferait fuir le futur dans le signal — c'est l'erreur classique qui
    fabrique des strategies magnifiques et fausses.
    """
    lows: List[int] = []
    highs: List[int] = []
    was_low = was_high = False

    for i in range(window, len(funding)):
        hist = sorted(r for _, r in funding[i - window:i])
        rate = funding[i][1]
        lo = hist[int(len(hist) * low_pct)]
        hi = hist[min(int(len(hist) * high_pct), len(hist) - 1)]

        is_low, is_high = rate <= lo, rate >= hi
        if is_low and not was_low:
            lows.append(funding[i][0])
        if is_high and not was_high:
            highs.append(funding[i][0])
        was_low, was_high = is_low, is_high

    return lows, highs


def events_to_indices(candles: Sequence[Candle], events: Sequence[int]) -> set:
    """Associe chaque evenement a la premiere bougie qui CLOTURE apres lui.

    Decisif pour l'honnetete du test : le taux de financement d'un instant
    donne n'est connu qu'a cet instant, donc le signal ne peut agir qu'a
    partir de la bougie suivante.
    """
    out = set()
    if not candles or not events:
        return out
    k = 0
    for i, c in enumerate(candles):
        while k < len(events) and events[k] <= c.ts:
            out.add(i)
            k += 1
        if k >= len(events):
            break
    return out
