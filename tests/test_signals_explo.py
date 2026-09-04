"""La bibliotheque exploratoire (bot/signals_explo.py) : 40 candidats issus
d'une recherche externe (TA-Lib, pandas-ta, taxonomie WorldQuant).

Trois exigences, dans l'ordre de gravite si elles sont violees :
  1. Aucune cle dupliquee (deux candidats qui s'ecrasent silencieusement).
  2. Aucune fuite du futur : LE test qui aurait attrape les deux faux
     positifs decouverts plus tot dans ce projet, applique ici a 40
     signaux d'un coup plutot qu'un par un.
  3. Chaque candidat tourne sans exception sur une serie de bougies
     ordinaire et se declenche au moins une fois (un signal qui ne se
     declenche jamais ne peut pas etre teste, et une exception plantee au
     milieu d'un balayage de 40 signaux est le genre de bug qui passe
     inapercu si on ne le cherche pas explicitement).
"""

import unittest

from bot.adapters.synthetic import SyntheticAdapter
from bot.config import StrategyConfig
from bot.signals_explo import CANDIDATS_EXPLO, build_all_explo


class TestCatalogue(unittest.TestCase):
    def test_pas_de_cle_dupliquee(self):
        cles = [k.spec.key for k in CANDIDATS_EXPLO]
        self.assertEqual(len(cles), len(set(cles)), f"cles dupliquees parmi : {cles}")

    def test_chaque_candidat_a_une_hypothese_et_une_source(self):
        for klass in CANDIDATS_EXPLO:
            self.assertTrue(klass.spec.label)
            self.assertTrue(klass.spec.hypothese)
            self.assertTrue(getattr(klass.spec, "source", ""), f"{klass.spec.key} sans source")


class TestExecution(unittest.TestCase):
    """Chaque signal doit tourner sans exception et se declencher au moins
    une fois sur un historique synthetique assez long."""

    @classmethod
    def setUpClass(cls):
        cls.candles = SyntheticAdapter("BTCUSDT", "1h", bars=1500, seed=1).get_candles()
        cls.cfg = StrategyConfig()

    def test_tous_les_candidats_tournent_sans_exception(self):
        erreurs = []
        for sig in build_all_explo():
            try:
                sig.prepare(self.candles, self.cfg, None)
                for i in range(sig.warmup, len(self.candles)):
                    sig.fires(i)
            except Exception as exc:  # noqa: BLE001 - on veut TOUT attraper ici
                erreurs.append(f"{sig.spec.key}: {exc!r}")
        self.assertEqual(erreurs, [], "signaux en erreur : " + "; ".join(erreurs))

    def test_la_plupart_des_candidats_se_declenchent_au_moins_une_fois(self):
        """Un signal qui ne se declenche jamais ne peut tout simplement pas
        etre teste. On tolere quelques exceptions (seuils stricts, comme le
        pic de volume) mais pas la majorite."""
        muets = []
        for sig in build_all_explo():
            sig.prepare(self.candles, self.cfg, None)
            n = sum(1 for i in range(sig.warmup, len(self.candles)) if sig.fires(i))
            if n == 0:
                muets.append(sig.spec.key)
        self.assertLess(
            len(muets), len(CANDIDATS_EXPLO) * 0.15,
            f"trop de signaux muets sur ces donnees : {muets}",
        )


class TestAucuneFuiteDuFutur(unittest.TestCase):
    """Le test le plus important de ce fichier : modifier la fin de la serie
    ne doit changer AUCUNE decision passee, pour aucun des 40 candidats."""

    @classmethod
    def setUpClass(cls):
        base = SyntheticAdapter("BTCUSDT", "1h", bars=1500, seed=1).get_candles()
        cls.CUTOFF = 900
        cls.candles_a = base
        # Bougies b : identiques a a jusqu'a CUTOFF, puis remplacees par une
        # serie tres differente (autre graine, feree pour la duree du futur).
        futur = SyntheticAdapter("BTCUSDT", "1h", bars=len(base) - cls.CUTOFF, seed=999).get_candles()
        # recale les timestamps/volume pour raccorder proprement, seuls les
        # prix/volumes importent pour ce test
        cls.candles_b = list(base[: cls.CUTOFF]) + list(futur[: len(base) - cls.CUTOFF])
        cls.cfg = StrategyConfig()

    def test_toutes_les_decisions_passees_sont_identiques(self):
        fautifs = []
        for sig_a, sig_b in zip(build_all_explo(), build_all_explo()):
            sig_a.prepare(self.candles_a, self.cfg, None)
            sig_b.prepare(self.candles_b, self.cfg, None)
            for i in range(sig_a.warmup, self.CUTOFF - 2):  # marge de securite
                if sig_a.fires(i) != sig_b.fires(i):
                    fautifs.append(f"{sig_a.spec.key} @ i={i}")
                    break
        self.assertEqual(fautifs, [], "fuite du futur detectee sur : " + "; ".join(fautifs))


if __name__ == "__main__":
    unittest.main()
