"""Adaptateur OANDA (forex) — donnees de marche, en lecture seule.

Pourquoi le forex apres l'echec sur BTC
----------------------------------------
Tous les candidats testes sur BTC ont ete tues par la meme contrainte :
l'avantage mesure etait plus petit que le cout de le capturer. Un
aller-retour en spot crypto coute environ 0,30 %.

Sur EUR/USD, il coute 0,02 a 0,03 % — dix a quinze fois moins. Le marche
y est PLUS efficient, donc les avantages y sont plus petits ; mais le
peage l'est bien davantage, et c'est ce rapport-la qui decide. Un
avantage de 0,18 %, sans valeur sur BTC, y serait tres rentable.

S'ajoutent deux proprietes que la crypto n'offrait pas : beaucoup de
donnees intraday sur un instrument unique (donc de la puissance
statistique sans dependre de paires correlees entre elles), et aucun
biais du survivant — EUR/USD n'a jamais ete deliste.

SECURITE
--------
Ce module n'implemente QUE la lecture de bougies. Aucune fonction de
passage d'ordre n'y existe, et il pointe par defaut sur l'environnement
« practice » (compte de demonstration a fonds fictifs). Le jeton est lu
dans la variable d'environnement OANDA_TOKEN et n'est jamais ecrit dans
un fichier de configuration — donc jamais commite par accident.

    export OANDA_TOKEN="ton-jeton-practice"

Le jeton s'obtient sur un compte demo OANDA, gratuit : rubrique « Gerer
les acces API », generation en quelques minutes.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import Candle
from .base import (
    MarketAdapter,
    MarketDataError,
    merge_candles,
    read_cache,
    sanity_check,
    write_cache,
)

HOSTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}

# Correspondance entre nos unites de temps et celles d'OANDA.
GRANULARITE = {
    "1m": "M1",
    "5m": "M5",
    "15m": "M15",
    "30m": "M30",
    "1h": "H1",
    "2h": "H2",
    "4h": "H4",
    "6h": "H6",
    "12h": "H12",
    "1d": "D",
}

MAX_COUNT = 5000  # plafond OANDA par requete


def parse_candles(payload: Dict[str, Any]) -> List[Candle]:
    """Convertit une reponse OANDA en bougies.

    Deux regles importantes :
      - seules les bougies `complete: true` sont retenues. Une bougie en
        cours de formation change encore de valeur ; l'inclure fausserait
        tout ce qui vient ensuite.
      - les prix utilises sont les prix MEDIANS (mid). Le cout du spread
        est modelise separement, via le slippage du portefeuille, plutot
        que melange aux donnees.
    """
    out: List[Candle] = []
    for c in payload.get("candles", []):
        if not c.get("complete"):
            continue
        mid = c.get("mid")
        if not mid:
            continue
        ts = _rfc3339_to_ms(c["time"])
        out.append(
            Candle(
                ts=ts,
                open=float(mid["o"]),
                high=float(mid["h"]),
                low=float(mid["l"]),
                close=float(mid["c"]),
                volume=float(c.get("volume", 0)),
            )
        )
    return out


_RFC3339 = re.compile(
    r"^(?P<base>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<frac>\d+))?"
    r"(?P<zone>Z|[+-]\d{2}:?\d{2})?$"
)


def _rfc3339_to_ms(texte: str) -> int:
    """OANDA horodate a la nanoseconde ; datetime n'accepte que 6 chiffres."""
    m = _RFC3339.match(texte.strip())
    if not m:
        raise MarketDataError(f"Horodatage OANDA illisible : {texte!r}")
    base = m.group("base").replace(" ", "T")
    frac = (m.group("frac") or "")[:6].ljust(6, "0")
    zone = m.group("zone") or "Z"
    zone = "+00:00" if zone == "Z" else zone
    dt = datetime.fromisoformat(f"{base}.{frac}{zone}")
    return int(dt.timestamp() * 1000)


def _ms_to_rfc3339(ms: int) -> str:
    return (
        datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    )


class OandaAdapter(MarketAdapter):
    name = "oanda"

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        cache_dir: str = "data",
        environnement: str = "practice",
        token: Optional[str] = None,
    ) -> None:
        super().__init__(symbol, timeframe)
        if timeframe not in GRANULARITE:
            raise MarketDataError(
                f"OANDA n'accepte que ces unites de temps : {', '.join(GRANULARITE)}"
            )
        if "_" not in symbol:
            raise MarketDataError(
                f"OANDA attend un instrument de la forme 'EUR_USD' (recu : {symbol})"
            )
        if environnement not in HOSTS:
            raise MarketDataError("environnement OANDA : 'practice' ou 'live'")

        self.granularite = GRANULARITE[timeframe]
        self.host = HOSTS[environnement]
        self.environnement = environnement
        self.token = token or os.environ.get("OANDA_TOKEN", "")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_file(self) -> Path:
        return self.cache_dir / f"oanda_{self.symbol}_{self.timeframe}.csv"

    # ------------------------------------------------------------------
    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.token:
            raise MarketDataError(
                "Aucun jeton OANDA. Cree un compte practice (gratuit), genere un "
                "jeton d'acces, puis :\n\n    export OANDA_TOKEN=\"ton-jeton\"\n\n"
                "Le jeton n'est jamais ecrit dans un fichier de configuration, "
                "pour qu'il ne parte jamais dans un depot git."
            )
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.host}/v3/instruments/{self.symbol}/candles?{query}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept-Datetime-Format": "RFC3339",
                "User-Agent": "paper-trading-bot/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise MarketDataError(
                    "OANDA refuse le jeton (401). Verifie qu'il correspond bien a "
                    f"l'environnement '{self.environnement}' : un jeton de compte "
                    "reel ne fonctionne pas sur practice, et inversement."
                ) from exc
            if exc.code == 400:
                raise MarketDataError(
                    f"OANDA rejette la requete (400). L'instrument '{self.symbol}' "
                    "existe-t-il ? Le format attendu est 'EUR_USD'."
                ) from exc
            raise MarketDataError(f"OANDA a repondu HTTP {exc.code} : {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise MarketDataError(f"Impossible de joindre OANDA ({exc.reason})") from exc

    def _fetch(self, start_ms: int, end_ms: int) -> List[Candle]:
        """Le forex ferme le week-end : les bougies ne sont pas contigues.

        On avance donc par nombre de bougies plutot que par duree, sinon
        chaque requete couvrant un week-end reviendrait vide et la
        pagination s'arreterait pretendument au bout de l'historique.
        """
        out: List[Candle] = []
        curseur = start_ms
        vides = 0
        while curseur < end_ms:
            payload = self._request(
                {
                    "granularity": self.granularite,
                    "price": "M",
                    "from": _ms_to_rfc3339(curseur),
                    "count": MAX_COUNT,
                }
            )
            lot = [c for c in parse_candles(payload) if c.ts <= end_ms]
            if not lot:
                vides += 1
                if vides >= 2:
                    break
                curseur += 7 * 86_400_000  # on saute une semaine et on reessaie
                continue
            vides = 0
            out.extend(lot)
            dernier = lot[-1].ts
            if dernier + self.interval_ms <= curseur:
                break
            curseur = dernier + self.interval_ms
            if len(lot) < MAX_COUNT // 2:
                break
            time.sleep(0.2)
        return out

    # ------------------------------------------------------------------
    def get_candles(self, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[Candle]:
        maintenant = int(time.time() * 1000)
        derniere_close = (maintenant // self.interval_ms) * self.interval_ms - self.interval_ms
        end_ms = min(end_ms or derniere_close, derniere_close)
        start_ms = start_ms or (end_ms - 730 * 86_400_000)

        cache = read_cache(self.cache_file)
        if cache and cache[0].ts <= start_ms and cache[-1].ts >= end_ms - self.interval_ms:
            return sanity_check([c for c in cache if start_ms <= c.ts <= end_ms])

        depuis = start_ms
        if cache and cache[0].ts <= start_ms <= cache[-1].ts:
            depuis = cache[-1].ts + self.interval_ms

        neuf = self._fetch(depuis, end_ms) if depuis <= end_ms else []
        tout = merge_candles(cache, neuf)
        if tout:
            write_cache(self.cache_file, tout)
        return [c for c in tout if start_ms <= c.ts <= end_ms]

    def get_price(self) -> float:
        payload = self._request(
            {"granularity": self.granularite, "price": "M", "count": 2}
        )
        bougies = parse_candles(payload)
        if not bougies:
            raise MarketDataError("OANDA n'a renvoye aucune bougie complete.")
        return bougies[-1].close
