"""Bibliotheque EXPLORATOIRE de signaux candidats (issue d'une recherche externe).

Pourquoi ce fichier est SEPARE de bot/signals.py
-------------------------------------------------
bot/signals.py contient la liste figee, pre-enregistree, de 5 signaux de
prix + 2 signaux de financement, testee methodiquement tout au long de ce
projet. Elle reste la reference "serieuse".

Ce fichier-ci est different par nature : les ~40 signaux ci-dessous sont
issus d'une recherche large sur des bibliotheques publiques (TA-Lib,
pandas-ta, la taxonomie des 101 alphas WorldQuant, la litterature
academique sur les facteurs momentum/reversion). Le catalogue complet,
avec sources et candidats pas encore code, est dans catalogue_signaux.md
(100+ entrees). Voir la aussi pour la justification de chaque candidat.

LE PROTOCOLE EN DEUX ETAPES (pourquoi tester 40+ signaux ne re-ouvre pas
la porte au p-hacking que ce projet a combattu depuis le debut)
--------------------------------------------------------------------------
Avec un seuil brut de 5 % et ~40 signaux x 5 horizons = ~200 tests, on
attend ENVIRON 10 "significatifs" par pur hasard, meme si rien ici ne
vaut quoi que ce soit. Une correction de Bonferroni sur 200 tests donne
un seuil de 0.05/200 = 0.00025 : tres strict, ce qui est voulu pour un
DEPISTAGE (stage 1), pas pour trancher.

Etape 1 - DEPISTAGE (`python -m bot.cli explore`) : tous ces candidats
sont testes sur le bloc TRAIN uniquement (le bloc VALIDATION reste
intact). On classe par p-value, sans pretendre qu'un p<0.05 brut prouve
quoi que ce soit — c'est un tri, pas un verdict.

Etape 2 - CONFIRMATION : seuls les 3 a 5 meilleurs de l'etape 1 sont
retestes UNE FOIS sur le bloc VALIDATION, avec :
    python -m bot.cli edge --signal <cle> --explo --segment validation
Le seuil corrige a cette etape se calcule sur la taille de la LISTE
COURTE (3-5 candidats), pas sur les 40 initiaux : c'est le depistage qui
absorbe le cout statistique d'avoir cherche large, la confirmation
verifie que le survivant n'etait pas juste le plus chanceux du lot.

Le bloc de RESERVE (holdout) n'intervient nulle part ici. Il reste reserve
au verdict final de la strategie complete, une fois figee — jamais a la
recherche de signaux.

REGLE IDENTIQUE A bot/signals.py, A PARTIR DE MAINTENANT
----------------------------------------------------------
Cette liste de candidats est CONNUE ET FIGEE au moment ou ce fichier est
ecrit, AVANT d'avoir vu le moindre resultat. On ne l'allonge pas apres
coup parce qu'un candidat "presque significatif" donnerait envie d'en
ajouter un neuvieme cousin. Un futur lot de candidats (voir
catalogue_signaux.md pour ceux pas encore codes) sera une nouvelle serie
pre-enregistree, avec sa propre correction — exactement comme la serie
financement l'a ete pour bot/signals.py.

Tous les signaux sont des EVENEMENTS (une transition), jamais un etat
permanent, pour la meme raison que dans bot/signals.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .config import StrategyConfig
from .indicators import (
    aroon,
    atr as atr_series,
    awesome_oscillator,
    bollinger,
    cci,
    cmf,
    crossed_above,
    crossed_up,
    ema,
    keltner,
    macd,
    mfi,
    obv,
    roc,
    rolling_max,
    rolling_min,
    rsi,
    sma,
    stochastic,
    trix,
    vortex,
    volume_sma,
    williams_r,
)
from .models import Candle
from .signals import Signal, SignalSpec


@dataclass
class SpecExplo(SignalSpec):
    """SignalSpec enrichi : categorie economique + source de l'idee.

    Reste compatible avec tout le code qui attend juste .key/.label/.hypothese
    (bot/edge.py, bot/pooled.py) puisque c'est une sous-classe de SignalSpec.
    """
    categorie: str = ""
    source: str = ""
    correle_avec: str = ""  # signal(s) proche(s), a garder en tete en lisant les resultats


# ----------------------------------------------------------------------
# Gabarits generiques : deux familles couvrent la quasi-totalite des
# signaux techniques usuels. Ecrire 40 classes a la main aurait duplique
# la meme logique 40 fois ; ecrire deux gabarits et 40 `_compute()` est a
# la fois plus court ET plus facile a auditer (une seule implementation
# du "quand est-ce qu'un evenement se declenche" a verifier).
# ----------------------------------------------------------------------
class SeuilVersLeHaut(Signal):
    """Gabarit : une serie calculee franchit un niveau fixe vers le haut."""

    spec: SpecExplo
    level: float = 0.0
    _warmup: int = 60

    def _compute(self, candles: Sequence[Candle], cfg: StrategyConfig) -> List[Optional[float]]:
        raise NotImplementedError

    def prepare(self, candles, cfg, extras=None):
        self.serie = self._compute(candles, cfg)

    def fires(self, i):
        return crossed_above(self.serie, self.level, i)

    @property
    def warmup(self):
        return self._warmup


class CroisementDeSeries(Signal):
    """Gabarit : une serie (rapide) franchit une autre (lente) vers le haut."""

    spec: SpecExplo
    _warmup: int = 60

    def _compute(self, candles: Sequence[Candle], cfg: StrategyConfig):
        raise NotImplementedError  # -> (rapide, lente)

    def prepare(self, candles, cfg, extras=None):
        self.rapide, self.lente = self._compute(candles, cfg)

    def fires(self, i):
        return crossed_up(self.rapide, self.lente, i)

    @property
    def warmup(self):
        return self._warmup


# ----------------------------------------------------------------------
# 1. Rebonds de survente (oscillateurs bornes)
# ----------------------------------------------------------------------
class StochRebond(SeuilVersLeHaut):
    spec = SpecExplo("stoch_rebond", "Stochastique lent sort de survente (%K > 20)",
                      "retour a la moyenne : le %K stochastique repasse au-dessus de 20",
                      categorie="rebond_survente", source="TA-Lib/pandas-ta: Stochastic Oscillator")
    level = 20.0
    _warmup = 20

    def _compute(self, candles, cfg):
        k, _ = stochastic(candles, 14, 3, 3)
        return k


class StochRapideRebond(SeuilVersLeHaut):
    spec = SpecExplo("stoch_rapide_rebond", "Stochastique rapide sort de survente (%K(5) > 20)",
                      "variante rapide du precedent : periode plus courte, plus de signaux, plus de bruit",
                      categorie="rebond_survente", source="TA-Lib: STOCHF",
                      correle_avec="stoch_rebond")
    level = 20.0
    _warmup = 15

    def _compute(self, candles, cfg):
        k, _ = stochastic(candles, 5, 3, 3)
        return k


class WilliamsRebond(SeuilVersLeHaut):
    spec = SpecExplo("williams_rebond", "Williams %R sort de survente (> -80)",
                      "retour a la moyenne, formule equivalente au stochastique inverse",
                      categorie="rebond_survente", source="TA-Lib: WILLR",
                      correle_avec="stoch_rebond (formule quasi identique)")
    level = -80.0
    _warmup = 20

    def _compute(self, candles, cfg):
        return williams_r(candles, 14)


class CciRebond(SeuilVersLeHaut):
    spec = SpecExplo("cci_rebond", "Commodity Channel Index sort de survente (> -100)",
                      "le prix typique s'ecarte fortement sous sa moyenne puis y revient",
                      categorie="rebond_survente", source="TA-Lib: CCI")
    level = -100.0
    _warmup = 25

    def _compute(self, candles, cfg):
        return cci(candles, 20)


class MfiRebond(SeuilVersLeHaut):
    spec = SpecExplo("mfi_rebond", "Money Flow Index sort de survente (> 20)",
                      "RSI pondere par le volume : la pression vendeuse epuise le volume, puis se retourne",
                      categorie="rebond_survente", source="TA-Lib: MFI")
    level = 20.0
    _warmup = 20

    def _compute(self, candles, cfg):
        return mfi(candles, 14)


class RsiCourtRebond(SeuilVersLeHaut):
    spec = SpecExplo("rsi_court_rebond", "RSI(9) sort de survente (> 25)",
                      "variante plus courte et plus stricte du rsi_rebond deja teste (periode 14, seuil 30)",
                      categorie="rebond_survente", source="pratique courante (RSI court terme)",
                      correle_avec="rsi_rebond (bot/signals.py)")
    level = 25.0
    _warmup = 15

    def _compute(self, candles, cfg):
        return rsi([c.close for c in candles], 9)


class RsiSurventeProfonde(SeuilVersLeHaut):
    spec = SpecExplo("rsi_survente_profonde", "RSI(14) sort d'une survente extreme (> 20)",
                      "hypothese distincte : seuls les exces VRAIMENT extremes (pas 30, mais 20) ont un avantage",
                      categorie="rebond_survente", source="variante de seuil, litterature retail",
                      correle_avec="rsi_rebond (bot/signals.py)")
    level = 20.0
    _warmup = 20

    def _compute(self, candles, cfg):
        return rsi([c.close for c in candles], 14)


class RocCourtZero(SeuilVersLeHaut):
    spec = SpecExplo("roc_court_zero", "Rate of Change(12) repasse positif",
                      "momentum brut a court terme, sans lissage moyenne mobile",
                      categorie="momentum", source="TA-Lib: ROC")
    level = 0.0
    _warmup = 15

    def _compute(self, candles, cfg):
        return roc([c.close for c in candles], 12)


class RocLongZero(SeuilVersLeHaut):
    spec = SpecExplo("roc_long_zero", "Rate of Change(25) repasse positif",
                      "meme logique que roc_court_zero mais sur un horizon plus lent",
                      categorie="momentum", source="TA-Lib: ROC",
                      correle_avec="roc_court_zero")
    level = 0.0
    _warmup = 28

    def _compute(self, candles, cfg):
        return roc([c.close for c in candles], 25)


# ----------------------------------------------------------------------
# 2. Franchissements de la ligne zero (indicateurs de tendance/momentum)
# ----------------------------------------------------------------------
class MacdHistZero(SeuilVersLeHaut):
    spec = SpecExplo("macd_hist_zero", "Histogramme MACD repasse positif",
                      "l'ecart entre la ligne MACD et son signal change de sens : acceleration du momentum",
                      categorie="momentum", source="TA-Lib/pandas-ta: MACD")
    level = 0.0
    _warmup = 40

    def _compute(self, candles, cfg):
        _, _, hist = macd([c.close for c in candles])
        return hist


class MacdLigneZero(SeuilVersLeHaut):
    spec = SpecExplo("macd_ligne_zero", "Ligne MACD repasse positive",
                      "la moyenne courte repasse au-dessus de la longue en valeur absolue, pas seulement l'une vs l'autre",
                      categorie="tendance", source="TA-Lib/pandas-ta: MACD",
                      correle_avec="macd_hist_zero, ema_cross_moyen")
    level = 0.0
    _warmup = 40

    def _compute(self, candles, cfg):
        line, _, _ = macd([c.close for c in candles])
        return line


class TrixZero(SeuilVersLeHaut):
    spec = SpecExplo("trix_zero", "TRIX repasse positif",
                      "taux de variation d'une triple EMA : filtre le bruit court terme mieux qu'une EMA simple",
                      categorie="tendance", source="TA-Lib: TRIX")
    level = 0.0
    _warmup = 50

    def _compute(self, candles, cfg):
        return trix([c.close for c in candles], 15)


class AwesomeOscZero(SeuilVersLeHaut):
    spec = SpecExplo("awesome_osc_zero", "Awesome Oscillator repasse positif",
                      "SMA(5)-SMA(34) du point milieu : momentum de tendance, insensible au bruit intra-bougie",
                      categorie="momentum", source="pandas-ta: Awesome Oscillator (Bill Williams)")
    level = 0.0
    _warmup = 36

    def _compute(self, candles, cfg):
        return awesome_oscillator(candles)


class CmfZero(SeuilVersLeHaut):
    spec = SpecExplo("cmf_zero", "Chaikin Money Flow repasse positif",
                      "pression acheteuse (cloture proche du haut, volume a l'appui) redevient dominante",
                      categorie="volume", source="TA-Lib/pandas-ta: Chaikin Money Flow")
    level = 0.0
    _warmup = 25

    def _compute(self, candles, cfg):
        return cmf(candles, 20)


class ObvRocZero(SeuilVersLeHaut):
    spec = SpecExplo("obv_roc_zero", "Taux de variation de l'On Balance Volume repasse positif",
                      "le volume cumule signe change de tendance avant (ou avec) le prix",
                      categorie="volume", source="TA-Lib: OBV + ROC compose")
    level = 0.0
    _warmup = 20

    def _compute(self, candles, cfg):
        return roc(obv(candles), 14)


# ----------------------------------------------------------------------
# 3. Croisements de deux series
# ----------------------------------------------------------------------
class MacdSignalCross(CroisementDeSeries):
    spec = SpecExplo("macd_signal_cross", "Ligne MACD croise au-dessus de son signal",
                      "la formulation la plus classique du MACD, distincte des versions 'ligne vs zero'",
                      categorie="tendance", source="TA-Lib/pandas-ta: MACD",
                      correle_avec="macd_hist_zero (les deux se declenchent presque ensemble)")
    _warmup = 40

    def _compute(self, candles, cfg):
        line, sig, _ = macd([c.close for c in candles])
        return line, sig


class StochKdCross(CroisementDeSeries):
    spec = SpecExplo("stoch_kd_cross", "%K stochastique croise au-dessus de %D",
                      "signal de retournement classique, different du simple seuil de survente",
                      categorie="rebond_survente", source="TA-Lib/pandas-ta: Stochastic",
                      correle_avec="stoch_rebond")
    _warmup = 20

    def _compute(self, candles, cfg):
        k, d = stochastic(candles, 14, 3, 3)
        return k, d


class EmaCrossCourt(CroisementDeSeries):
    spec = SpecExplo("ema_cross_court", "EMA(5) croise au-dessus de EMA(20)",
                      "suivi de tendance tres reactif, variante courte du croisement d'EMA deja teste",
                      categorie="tendance", source="pratique courante",
                      correle_avec="ema_cross (bot/signals.py)")
    _warmup = 22

    def _compute(self, candles, cfg):
        closes = [c.close for c in candles]
        return ema(closes, 5), ema(closes, 20)


class EmaCrossMoyen(CroisementDeSeries):
    spec = SpecExplo("ema_cross_moyen", "EMA(10) croise au-dessus de EMA(50)",
                      "suivi de tendance a horizon intermediaire",
                      categorie="tendance", source="pratique courante",
                      correle_avec="ema_cross (bot/signals.py), ema_cross_court")
    _warmup = 52

    def _compute(self, candles, cfg):
        closes = [c.close for c in candles]
        return ema(closes, 10), ema(closes, 50)


class EmaCrossLong(CroisementDeSeries):
    spec = SpecExplo("ema_cross_long", "EMA(50) croise au-dessus de EMA(200) (« golden cross »)",
                      "le croisement de suivi de tendance le plus etudie dans la litterature retail",
                      categorie="tendance", source="pratique courante (golden/death cross)",
                      correle_avec="ema_cross (bot/signals.py)")
    _warmup = 205

    def _compute(self, candles, cfg):
        closes = [c.close for c in candles]
        return ema(closes, 50), ema(closes, 200)


class SmaCross(CroisementDeSeries):
    spec = SpecExplo("sma_cross", "SMA(20) croise au-dessus de SMA(50)",
                      "meme logique que le croisement d'EMA, avec un lissage different (SMA non pondere)",
                      categorie="tendance", source="pratique courante",
                      correle_avec="ema_cross_moyen")
    _warmup = 52

    def _compute(self, candles, cfg):
        closes = [c.close for c in candles]
        return sma(closes, 20), sma(closes, 50)


class VortexCross(CroisementDeSeries):
    spec = SpecExplo("vortex_cross", "VI+ croise au-dessus de VI-",
                      "mouvement directionnel haussier depasse le mouvement directionnel baissier",
                      categorie="tendance", source="TA-Lib/pandas-ta: Vortex Indicator")
    _warmup = 18

    def _compute(self, candles, cfg):
        plus, minus = vortex(candles, 14)
        return plus, minus


class AroonCross(CroisementDeSeries):
    spec = SpecExplo("aroon_cross", "Aroon Up croise au-dessus de Aroon Down",
                      "le plus haut recent redevient plus proche que le plus bas recent",
                      categorie="tendance", source="TA-Lib/pandas-ta: Aroon")
    _warmup = 18

    def _compute(self, candles, cfg):
        up, down = aroon(candles, 14)
        return up, down


# ----------------------------------------------------------------------
# 4. Cassures de canal (close franchit une bande haute)
# ----------------------------------------------------------------------
class BollingerCassure20(CroisementDeSeries):
    spec = SpecExplo("bollinger_cassure_20", "Cloture franchit la bande de Bollinger haute (20, 2σ)",
                      "momentum reel : le prix sort de son enveloppe de volatilite habituelle",
                      categorie="cassure", source="TA-Lib/pandas-ta: Bollinger Bands",
                      correle_avec="cassure (bot/signals.py)")
    _warmup = 22

    def _compute(self, candles, cfg):
        closes = [c.close for c in candles]
        _, upper, _ = bollinger(closes, 20, 2.0)
        return closes, upper


class BollingerCassure10(CroisementDeSeries):
    spec = SpecExplo("bollinger_cassure_10", "Cloture franchit la bande de Bollinger haute (10, 2.5σ)",
                      "variante plus etroite/rapide de bollinger_cassure_20",
                      categorie="cassure", source="TA-Lib/pandas-ta: Bollinger Bands",
                      correle_avec="bollinger_cassure_20")
    _warmup = 12

    def _compute(self, candles, cfg):
        closes = [c.close for c in candles]
        _, upper, _ = bollinger(closes, 10, 2.5)
        return closes, upper


class KeltnerCassure(CroisementDeSeries):
    spec = SpecExplo("keltner_cassure", "Cloture franchit le canal de Keltner haut",
                      "meme idee que Bollinger, mais l'enveloppe est batie sur l'ATR plutot que l'ecart-type",
                      categorie="cassure", source="TA-Lib/pandas-ta: Keltner Channel",
                      correle_avec="bollinger_cassure_20")
    _warmup = 22

    def _compute(self, candles, cfg):
        closes = [c.close for c in candles]
        _, upper, _ = keltner(candles, 20, 10, 2.0)
        return closes, upper


class DonchianCassure20(CroisementDeSeries):
    spec = SpecExplo("donchian_cassure_20", "Cassure du plus haut des 20 dernieres bougies",
                      "variante plus courte de la cassure deja testee (48 bougies) : plus de signaux, plus de bruit",
                      categorie="cassure", source="TA-Lib: variante Donchian",
                      correle_avec="cassure (bot/signals.py)")
    _warmup = 22

    def _compute(self, candles, cfg):
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        return closes, rolling_max(highs, 20)


class DonchianCassure96(CroisementDeSeries):
    spec = SpecExplo("donchian_cassure_96", "Cassure du plus haut des 96 dernieres bougies",
                      "variante plus longue : moins de signaux, cassures plus significatives",
                      categorie="cassure", source="TA-Lib: variante Donchian",
                      correle_avec="cassure (bot/signals.py)")
    _warmup = 98

    def _compute(self, candles, cfg):
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        return closes, rolling_max(highs, 96)


class DonchianCassure200(CroisementDeSeries):
    spec = SpecExplo("donchian_cassure_200", "Cassure du plus haut des 200 dernieres bougies",
                      "horizon multi-mois : capture les cassures de range majeures uniquement",
                      categorie="cassure", source="TA-Lib: variante Donchian",
                      correle_avec="cassure (bot/signals.py)")
    _warmup = 202

    def _compute(self, candles, cfg):
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        return closes, rolling_max(highs, 200)


# ----------------------------------------------------------------------
# 5. Rebonds sur bande basse (retour a la moyenne, cote oppose de la cassure)
# ----------------------------------------------------------------------
class BollingerRebondBas(CroisementDeSeries):
    spec = SpecExplo("bollinger_rebond_bas", "Cloture rentre dans la bande apres etre passee sous la bande basse",
                      "exces baissier statistique : le prix revient dans son enveloppe habituelle",
                      categorie="retour_moyenne", source="TA-Lib/pandas-ta: Bollinger Bands",
                      correle_avec="creux (bot/signals.py)")
    _warmup = 22

    def _compute(self, candles, cfg):
        closes = [c.close for c in candles]
        _, _, lower = bollinger(closes, 20, 2.0)
        return closes, lower


class KeltnerRebondBas(CroisementDeSeries):
    spec = SpecExplo("keltner_rebond_bas", "Cloture rentre dans le canal apres etre passee sous le canal bas",
                      "meme idee que bollinger_rebond_bas avec une enveloppe batie sur l'ATR",
                      categorie="retour_moyenne", source="TA-Lib/pandas-ta: Keltner Channel",
                      correle_avec="bollinger_rebond_bas, creux (bot/signals.py)")
    _warmup = 22

    def _compute(self, candles, cfg):
        closes = [c.close for c in candles]
        _, _, lower = keltner(candles, 20, 10, 2.0)
        return closes, lower


class DonchianRebondBas(CroisementDeSeries):
    spec = SpecExplo("donchian_rebond_bas", "Cloture rentre au-dessus du plus bas des 20 dernieres bougies",
                      "rebond sur un support recent, apres l'avoir teste par en dessous",
                      categorie="retour_moyenne", source="TA-Lib: variante Donchian",
                      correle_avec="creux (bot/signals.py)")
    _warmup = 22

    def _compute(self, candles, cfg):
        closes = [c.close for c in candles]
        lows = [c.low for c in candles]
        return closes, rolling_min(lows, 20)


# ----------------------------------------------------------------------
# 6. Signaux bespoke (logique propre, pas reductible aux deux gabarits)
# ----------------------------------------------------------------------
class VolumeSpikeHaussier(Signal):
    spec = SpecExplo("volume_spike_haussier", "Pic de volume (>2x la moyenne 20) sur une bougie haussiere",
                      "un volume anormal a l'appui d'une hausse trahit un interet institutionnel",
                      categorie="volume", source="pratique courante (volume breakout)")

    def prepare(self, candles, cfg, extras=None):
        self.vol = [c.volume for c in candles]
        self.vsma = volume_sma(candles, 20)
        self.closes = [c.close for c in candles]
        self.opens = [c.open for c in candles]

    def fires(self, i):
        if i < 1:
            return False
        vs = self.vsma[i]
        if vs is None or vs <= 0:
            return False
        return self.vol[i] > 2.0 * vs and self.closes[i] > self.opens[i]

    @property
    def warmup(self):
        return 22


class DivergenceObvHaussiere(Signal):
    spec = SpecExplo("divergence_obv_haussiere", "Le prix fait un nouveau plus bas mais l'OBV non",
                      "divergence haussiere classique : le volume cumule ne confirme pas la baisse du prix, "
                      "signe d'accumulation silencieuse",
                      categorie="volume", source="analyse technique classique (divergence prix/volume)",
                      correle_avec="creux (bot/signals.py)")
    window = 20

    def prepare(self, candles, cfg, extras=None):
        self.closes = [c.close for c in candles]
        self.rmin_close = rolling_min(self.closes, self.window)
        self.obv = obv(candles)
        self.robv_min = rolling_min(self.obv, self.window)

    def _divergence(self, i):
        rc, ro = self.rmin_close[i], self.robv_min[i]
        if rc is None or ro is None:
            return None
        return self.closes[i] < rc and self.obv[i] >= ro

    def fires(self, i):
        if i < 1:
            return False
        now, before = self._divergence(i), self._divergence(i - 1)
        return now is True and before is not True

    @property
    def warmup(self):
        return self.window + 2


class ConfirmationTendanceMomentum(Signal):
    spec = SpecExplo("confirmation_tendance_momentum", "RSI(14) traverse 50 pendant que EMA(20) > EMA(50)",
                      "signal compose : le momentum se retourne A L'INTERIEUR d'une tendance haussiere deja "
                      "en place, plutot que contre elle",
                      categorie="composite", source="pratique courante (filtre de tendance + momentum)",
                      correle_avec="ema_cross_moyen")

    def prepare(self, candles, cfg, extras=None):
        closes = [c.close for c in candles]
        self.ema20 = ema(closes, 20)
        self.ema50 = ema(closes, 50)
        self.rsi14 = rsi(closes, 14)

    def fires(self, i):
        if not crossed_above(self.rsi14, 50.0, i):
            return False
        e20, e50 = self.ema20[i], self.ema50[i]
        return e20 is not None and e50 is not None and e20 > e50

    @property
    def warmup(self):
        return 52


class CassureAvecVolume(Signal):
    spec = SpecExplo("cassure_avec_volume", "Cassure de plus haut (48 bougies) confirmee par un volume >1.5x la moyenne",
                      "filtre la cassure deja testee (bot/signals.py) par une confirmation de volume : "
                      "une cassure sans volume est souvent un faux signal",
                      categorie="composite", source="pratique courante (breakout + volume)",
                      correle_avec="cassure (bot/signals.py)")
    window = 48

    def prepare(self, candles, cfg, extras=None):
        self.high_max = rolling_max([c.high for c in candles], self.window)
        self.closes = [c.close for c in candles]
        self.vol = [c.volume for c in candles]
        self.vsma = volume_sma(candles, 20)

    def fires(self, i):
        if i < 1 or self.high_max[i] is None or self.high_max[i - 1] is None:
            return False
        cassure = self.closes[i] > self.high_max[i] and self.closes[i - 1] <= self.high_max[i - 1]
        if not cassure:
            return False
        vs = self.vsma[i]
        return vs is not None and vs > 0 and self.vol[i] > 1.5 * vs

    @property
    def warmup(self):
        return self.window + 2


class TripleConfirmation(Signal):
    spec = SpecExplo("triple_confirmation", "MACD haussier + %K>%D + cloture au-dessus de l'EMA(50), simultanement",
                      "trois familles d'indicateurs independantes (momentum, oscillateur, tendance) qui "
                      "s'accordent en meme temps : idee de vote/consensus a petite echelle",
                      categorie="composite", source="approche 'confluence' courante en analyse technique",
                      correle_avec="macd_signal_cross, stoch_kd_cross")

    def prepare(self, candles, cfg, extras=None):
        closes = [c.close for c in candles]
        _, _, self.hist = macd(closes)
        self.k, self.d = stochastic(candles, 14, 3, 3)
        self.ema50 = ema(closes, 50)
        self.closes = closes

    def _tous_verts(self, i):
        h, k, d, e, c = self.hist[i], self.k[i], self.d[i], self.ema50[i], self.closes[i]
        if None in (h, k, d, e):
            return None
        return h > 0 and k > d and c > e

    def fires(self, i):
        if i < 1:
            return False
        now, before = self._tous_verts(i), self._tous_verts(i - 1)
        return now is True and before is not True

    @property
    def warmup(self):
        return 52


class CompressionExpansion(Signal):
    spec = SpecExplo(
        "compression_expansion",
        "Bande de Bollinger au plus etroit depuis 100 bougies, puis cassure de la bande haute",
        "la volatilite se contracte avant de se relacher (« squeeze ») ; la cassure qui suit une "
        "compression serait plus fiable qu'une cassure ordinaire",
        categorie="volatilite", source="pratique courante (Bollinger Band Squeeze / TTM Squeeze)",
        correle_avec="bollinger_cassure_20",
    )
    fenetre_percentile = 100

    def prepare(self, candles, cfg, extras=None):
        closes = [c.close for c in candles]
        mid, upper, lower = bollinger(closes, 20, 2.0)
        self.closes = closes
        self.upper = upper
        self.largeur = [
            (u - l) / m if (u is not None and l is not None and m) else None
            for u, l, m in zip(upper, lower, mid)
        ]

    def _compresse(self, i):
        f = self.fenetre_percentile
        if i < f + 1:
            return None
        fenetre = self.largeur[i - f:i]
        if any(v is None for v in fenetre) or self.largeur[i - 1] is None:
            return None
        return self.largeur[i - 1] <= min(fenetre)

    def fires(self, i):
        if i < self.fenetre_percentile + 2:
            return False
        if not self._compresse(i):
            return False
        u0, u1 = self.upper[i - 1], self.upper[i]
        c0, c1 = self.closes[i - 1], self.closes[i]
        if u0 is None or u1 is None:
            return False
        return c0 <= u0 and c1 > u1

    @property
    def warmup(self):
        return self.fenetre_percentile + 25


class CompressionExpansionAtr(Signal):
    spec = SpecExplo(
        "compression_expansion_atr",
        "ATR relatif au plus bas depuis 100 bougies, puis bougie haussiere forte (>1.5x l'ATR)",
        "variante de compression_expansion basee sur l'ATR plutot que la largeur de Bollinger : "
        "la volatilite realisee (pas seulement l'ecart-type des clotures) se contracte puis explose",
        categorie="volatilite", source="pratique courante (volatility contraction pattern)",
        correle_avec="compression_expansion",
    )
    fenetre_percentile = 100

    def prepare(self, candles, cfg, extras=None):
        self.closes = [c.close for c in candles]
        self.opens = [c.open for c in candles]
        a = atr_series(candles, 14)
        self.atr_rel = [
            (v / c.close) if v is not None and c.close else None
            for v, c in zip(a, candles)
        ]
        self.atr = a

    def fires(self, i):
        f = self.fenetre_percentile
        if i < f + 15:
            return False
        fenetre = self.atr_rel[i - f:i]
        if any(v is None for v in fenetre) or self.atr_rel[i - 1] is None:
            return False
        etait_comprime = self.atr_rel[i - 1] <= min(fenetre)
        if not etait_comprime:
            return False
        a = self.atr[i]
        if a is None:
            return False
        bougie_forte = (self.closes[i] - self.opens[i]) > 1.5 * a
        return bougie_forte

    @property
    def warmup(self):
        return self.fenetre_percentile + 20


class ClotureForteApresRepli(SeuilVersLeHaut):
    spec = SpecExplo(
        "cloture_force_range",
        "La cloture capture >70% de l'amplitude haut-bas de la bougie (force acheteuse intra-bougie)",
        "inspire de la famille 'structure intra-journaliere' des alphas WorldQuant : une cloture proche "
        "du plus haut de la bougie trahit une pression acheteuse en fin de periode",
        categorie="momentum", source="WorldQuant 101 Alphas (famille intraday structure, adapte en mono-actif)",
    )
    level = 0.70
    _warmup = 5

    def _compute(self, candles, cfg):
        out: List[Optional[float]] = []
        for c in candles:
            span = c.high - c.low
            out.append(0.5 if span <= 0 else (c.close - c.low) / span)
        return out


# ----------------------------------------------------------------------
# Liste FIGEE des candidats exploratoires. Meme regle que dans
# bot/signals.py : on ne l'allonge pas apres avoir vu des resultats. Un
# lot supplementaire (voir catalogue_signaux.md) sera une nouvelle serie
# pre-enregistree distincte, avec sa propre correction.
# ----------------------------------------------------------------------
CANDIDATS_EXPLO: List[Callable[[], Signal]] = [
    # 1. rebonds de survente
    StochRebond, StochRapideRebond, WilliamsRebond, CciRebond, MfiRebond,
    RsiCourtRebond, RsiSurventeProfonde, RocCourtZero, RocLongZero,
    # 2. franchissements de zero
    MacdHistZero, MacdLigneZero, TrixZero, AwesomeOscZero, CmfZero, ObvRocZero,
    # 3. croisements de deux series
    MacdSignalCross, StochKdCross, EmaCrossCourt, EmaCrossMoyen, EmaCrossLong,
    SmaCross, VortexCross, AroonCross,
    # 4. cassures de canal
    BollingerCassure20, BollingerCassure10, KeltnerCassure,
    DonchianCassure20, DonchianCassure96, DonchianCassure200,
    # 5. rebonds sur bande basse
    BollingerRebondBas, KeltnerRebondBas, DonchianRebondBas,
    # 6. bespoke / composites / volatilite
    VolumeSpikeHaussier, DivergenceObvHaussiere, ConfirmationTendanceMomentum,
    CassureAvecVolume, TripleConfirmation, CompressionExpansion,
    CompressionExpansionAtr, ClotureForteApresRepli,
]


def build_all_explo() -> List[Signal]:
    return [klass() for klass in CANDIDATS_EXPLO]


def by_key(key: str) -> Signal:
    for klass in CANDIDATS_EXPLO:
        if klass.spec.key == key:
            return klass()
    dispo = ", ".join(k.spec.key for k in CANDIDATS_EXPLO)
    raise ValueError(f"Signal exploratoire inconnu : {key}. Disponibles : {dispo}")
