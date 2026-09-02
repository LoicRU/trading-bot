"""Adaptateur CSV : rejoue un historique telecharge a la main.

Utile quand le courtier n'expose aucune API — le cas d'OANDA TMS, qui ne
propose que MetaTrader 5 — quand l'API de l'exchange est bloquee, ou pour
exploiter une source d'historique profond.

Trois formats reconnus automatiquement
---------------------------------------
1. Simple, avec en-tete : ts,open,high,low,close,volume
   ts en millisecondes UTC, en secondes, ou en date ISO.

2. Export MetaTrader 5, reconnu a son en-tete entre chevrons :
   <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>

3. HistData « Generic ASCII », sans en-tete, separe par des points-virgules :
   20240102 000000;1.104370;1.104510;1.104110;1.104390;0
   Attention : ces donnees sont horodatees en EST sans heure d'ete, donc
   UTC-5 toute l'annee. D'ou le reglage market.csv_tz_offset_hours.

Lecture en flux
---------------
Vingt-cinq ans de M1 representent environ neuf millions de lignes. Les
charger toutes en memoire avant de les agreger demanderait plusieurs
gigaoctets pour rien. Le fichier est donc lu ligne par ligne et agrege au
vol : seules les bougies finales sont conservees — une centaine de
milliers en H1, quelques dizaines de megaoctets.

Le chemin peut designer un DOSSIER : tous les fichiers qu'il contient sont
lus et fusionnes. HistData livre un fichier par mois, ca evite d'avoir a
les concatener a la main.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from ..models import Candle
from .base import MarketAdapter, MarketDataError, sanity_check

# (ts_ms, open, high, low, close, volume)
Ligne = Tuple[int, float, float, float, float, float]

_HISTDATA = re.compile(r"^\s*(\d{8})\s+(\d{6})\s*[;,]")
EXTENSIONS = (".csv", ".txt")


# ----------------------------------------------------------------------
# Horodatages
# ----------------------------------------------------------------------
def _parse_ts(raw: str) -> int:
    raw = raw.strip()
    try:
        value = float(raw)
        return int(value * 1000) if value < 1e11 else int(value)
    except ValueError:
        pass
    text = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise MarketDataError(f"Horodatage illisible dans le CSV : {raw!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _parse_mt5_datetime(date_txt: str, time_txt: str) -> int:
    texte = f"{date_txt.strip().replace('.', '-')} {time_txt.strip()}"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return int(datetime.strptime(texte, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
    raise MarketDataError(f"Horodatage MetaTrader illisible : {date_txt!r} {time_txt!r}")


def _jours_depuis_epoch(y: int, m: int, d: int) -> int:
    """Nombre de jours entre 1970-01-01 et la date donnee, par calcul direct.

    Algorithme calendaire classique (Howard Hinnant), sans passer par
    datetime. La raison est purement pratique : sur vingt-cinq ans de
    donnees a la minute, soit environ neuf millions de lignes,
    strptime represente a lui seul plusieurs minutes d'attente. Ce calcul
    est une centaine de fois plus rapide, et un test verifie qu'il donne
    exactement les memes resultats que strptime.
    """
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _parse_histdata_datetime(date_txt: str, time_txt: str) -> int:
    y = int(date_txt[0:4]); mo = int(date_txt[4:6]); d = int(date_txt[6:8])
    h = int(time_txt[0:2]); mi = int(time_txt[2:4]); sec = int(time_txt[4:6])
    jours = _jours_depuis_epoch(y, mo, d)
    return (jours * 86400 + h * 3600 + mi * 60 + sec) * 1000


# ----------------------------------------------------------------------
# Detection de format
# ----------------------------------------------------------------------
def est_export_mt5(premiere_ligne: str) -> bool:
    tete = premiere_ligne.upper()
    return "<DATE>" in tete or ("<OPEN>" in tete and "<CLOSE>" in tete)


def est_histdata(premiere_ligne: str) -> bool:
    return bool(_HISTDATA.match(premiere_ligne))


# ----------------------------------------------------------------------
# Lecteurs en flux
# ----------------------------------------------------------------------
def _lignes_mt5(chemin: Path) -> Iterator[Ligne]:
    with chemin.open(encoding="utf-8-sig", newline="") as fh:
        entete = fh.readline()
        colonnes = [c.strip().strip("<>").upper() for c in entete.replace(",", "\t").split("\t")]
        index = {nom: k for k, nom in enumerate(colonnes) if nom}
        manquantes = [c for c in ("DATE", "OPEN", "HIGH", "LOW", "CLOSE") if c not in index]
        if manquantes:
            raise MarketDataError(f"Export MetaTrader : colonnes manquantes {manquantes}")

        for ligne in fh:
            champs = [c for c in (x.strip() for x in ligne.replace(",", "\t").split("\t")) if c]
            if len(champs) < 5:
                continue
            try:
                heure = champs[index["TIME"]] if "TIME" in index else "00:00:00"
                volume = 0.0
                for cle in ("TICKVOL", "VOLUME", "VOL"):
                    if cle in index and index[cle] < len(champs):
                        try:
                            volume = float(champs[index[cle]])
                            break
                        except ValueError:
                            pass
                yield (
                    _parse_mt5_datetime(champs[index["DATE"]], heure),
                    float(champs[index["OPEN"]]), float(champs[index["HIGH"]]),
                    float(champs[index["LOW"]]), float(champs[index["CLOSE"]]), volume,
                )
            except (ValueError, IndexError, MarketDataError):
                continue


def _lignes_histdata(chemin: Path) -> Iterator[Ligne]:
    with chemin.open(encoding="utf-8-sig", newline="") as fh:
        for ligne in fh:
            m = _HISTDATA.match(ligne)
            if not m:
                continue
            champs = ligne.rstrip().split(";")
            if len(champs) < 5:
                champs = ligne.rstrip().split(",")
            if len(champs) < 5:
                continue
            try:
                yield (
                    _parse_histdata_datetime(m.group(1), m.group(2)),
                    float(champs[1]), float(champs[2]),
                    float(champs[3]), float(champs[4]),
                    float(champs[5]) if len(champs) > 5 and champs[5] else 0.0,
                )
            except (ValueError, IndexError):
                continue


def _lignes_simple(chemin: Path) -> Iterator[Ligne]:
    with chemin.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise MarketDataError("CSV vide ou sans en-tete.")
        cols = {c.lower().strip(): c for c in reader.fieldnames}
        manquantes = [c for c in ("ts", "open", "high", "low", "close") if c not in cols]
        if manquantes:
            raise MarketDataError(
                f"Colonnes manquantes dans {chemin.name} : {manquantes}. "
                "Attendu : ts,open,high,low,close,volume"
            )
        for row in reader:
            try:
                vol = row[cols["volume"]] if "volume" in cols else ""
                yield (
                    _parse_ts(row[cols["ts"]]),
                    float(row[cols["open"]]), float(row[cols["high"]]),
                    float(row[cols["low"]]), float(row[cols["close"]]),
                    float(vol) if vol else 0.0,
                )
            except (ValueError, KeyError, TypeError):
                continue


def lignes_du_fichier(chemin: Path) -> Iterator[Ligne]:
    with chemin.open(encoding="utf-8-sig") as fh:
        premiere = fh.readline()
    if est_export_mt5(premiere):
        return _lignes_mt5(chemin)
    if est_histdata(premiere):
        return _lignes_histdata(chemin)
    return _lignes_simple(chemin)


def fichiers_de(chemin: Path) -> List[Path]:
    """Un fichier, ou tous les fichiers d'un dossier (HistData livre par mois)."""
    if chemin.is_dir():
        trouves = sorted(
            p for p in chemin.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        if not trouves:
            raise MarketDataError(
                f"Aucun fichier {' ou '.join(EXTENSIONS)} dans le dossier {chemin}"
            )
        return trouves
    return [chemin]


# ----------------------------------------------------------------------
# Agregation en flux
# ----------------------------------------------------------------------
def agreger_flux(lignes: Iterable[Ligne], interval_ms: int) -> List[Candle]:
    """Regroupe des bougies fines sans jamais toutes les garder en memoire.

    Chaque periode ne retient que sept nombres, quel que soit le nombre de
    bougies sources qu'elle contient. L'ordre des lignes n'a pas
    d'importance : on suit le premier et le dernier horodatage vus pour
    fixer correctement l'ouverture et la cloture.
    """
    seaux: Dict[int, List[float]] = {}
    for ts, o, h, l, c, v in lignes:
        cle = (ts // interval_ms) * interval_ms
        s = seaux.get(cle)
        if s is None:
            seaux[cle] = [o, h, l, c, v, ts, ts]
            continue
        s[1] = max(s[1], h)
        s[2] = min(s[2], l)
        s[4] += v
        if ts < s[5]:
            s[0], s[5] = o, ts
        if ts > s[6]:
            s[3], s[6] = c, ts

    return [
        Candle(ts=cle, open=s[0], high=s[1], low=s[2], close=s[3], volume=s[4])
        for cle, s in sorted(seaux.items())
    ]


def intervalle_dominant(ts_list: Sequence[int]) -> int:
    """Ecart le plus frequent entre horodatages consecutifs.

    Le mode et non la moyenne : le forex ferme le week-end, et quelques
    ecarts de 48 heures fausseraient completement une moyenne.
    """
    if len(ts_list) < 3:
        return 0
    compte: Dict[int, int] = {}
    for a, b in zip(ts_list, ts_list[1:]):
        d = b - a
        if d > 0:
            compte[d] = compte.get(d, 0) + 1
    return max(compte, key=compte.get) if compte else 0


class CsvAdapter(MarketAdapter):
    name = "csv"

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        csv_path: str,
        tz_offset_hours: float = 0.0,
    ) -> None:
        super().__init__(symbol, timeframe)
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise MarketDataError(f"Chemin CSV introuvable : {self.csv_path}")
        # Decalage horaire de la SOURCE par rapport a UTC (HistData : -5).
        self.tz_offset_ms = int(tz_offset_hours * 3_600_000)
        self._cache: Optional[List[Candle]] = None
        self.source_interval_ms = 0
        self.agrege = False
        self.nb_fichiers = 0
        self.fichiers_ignores: List[str] = []

    # ------------------------------------------------------------------
    def _flux(self) -> Iterator[Ligne]:
        """Un fichier illisible est ignore, pas fatal.

        Les archives HistData contiennent un fichier de licence en .txt a
        cote des donnees. Faire echouer tout le chargement a cause de lui
        serait absurde : on le saute et on le signale."""
        self.fichiers_ignores = []
        for fichier in fichiers_de(self.csv_path):
            try:
                for ts, o, h, l, c, v in lignes_du_fichier(fichier):
                    yield (ts - self.tz_offset_ms, o, h, l, c, v)
            except MarketDataError:
                self.fichiers_ignores.append(fichier.name)

    def _echantillon_interval(self) -> int:
        ts_list: List[int] = []
        for ligne in self._flux():
            ts_list.append(ligne[0])
            if len(ts_list) >= 200:
                break
        return intervalle_dominant(sorted(ts_list))

    def _load(self) -> List[Candle]:
        if self._cache is not None:
            return self._cache

        fichiers = fichiers_de(self.csv_path)
        self.nb_fichiers = len(fichiers)
        self.source_interval_ms = self._echantillon_interval()

        if self.source_interval_ms and self.source_interval_ms > self.interval_ms:
            raise MarketDataError(
                f"Le fichier contient des bougies de "
                f"{self.source_interval_ms // 60000} minutes, plus larges que l'unite "
                f"demandee ({self.timeframe}). On ne peut pas fabriquer du detail qui "
                "n'existe pas : reexporte dans une unite plus fine."
            )

        if self.source_interval_ms and self.source_interval_ms < self.interval_ms:
            candles = agreger_flux(self._flux(), self.interval_ms)
            self.agrege = True
        else:
            candles = [
                Candle(ts=ts, open=o, high=h, low=l, close=c, volume=v)
                for ts, o, h, l, c, v in self._flux()
            ]
            candles.sort(key=lambda c: c.ts)

        candles = sanity_check(candles)
        if not candles:
            raise MarketDataError(f"Aucune bougie exploitable dans {self.csv_path}")
        self._cache = candles
        return candles

    def get_candles(self, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[Candle]:
        candles = self._load()
        lo = start_ms if start_ms is not None else -1
        hi = end_ms if end_ms is not None else 1 << 62
        return [c for c in candles if lo <= c.ts <= hi]

    def get_price(self) -> float:
        candles = self._load()
        if not candles:
            raise MarketDataError("Aucune bougie dans le CSV.")
        return candles[-1].close

    def describe(self) -> str:
        base = f"csv:{self.symbol}@{self.timeframe}"
        details = []
        if self.nb_fichiers > 1:
            details.append(f"{self.nb_fichiers} fichiers")
        if self.agrege:
            details.append(f"agrege depuis {self.source_interval_ms // 60000} min")
        if self.tz_offset_ms:
            details.append(f"source UTC{self.tz_offset_ms / 3_600_000:+.0f}")
        return base + (f" ({', '.join(details)})" if details else "")
