"""Lecture des exports MetaTrader et HistData, et agregation M1 -> H1.

L'agregation est un endroit ou une erreur passe inapercue : des bougies
horaires fausses produisent des indicateurs faux, donc des signaux faux,
sans jamais lever d'exception. D'ou ces tests sur des valeurs connues.
"""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from bot.adapters.base import MarketDataError
from bot.adapters.csv_file import (
    CsvAdapter,
    agreger_flux,
    est_export_mt5,
    est_histdata,
    lignes_du_fichier,
)

ENTETE_MT5 = "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"


def ecrire(lignes, entete=""):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    if entete:
        f.write(entete)
    for l in lignes:
        f.write(l + "\n")
    f.close()
    return f.name


def mt5(date, heure, o, h, l, c, vol=100):
    return f"{date}\t{heure}\t{o}\t{h}\t{l}\t{c}\t{vol}\t0\t1"


def histdata(date, heure, o, h, l, c):
    return f"{date} {heure};{o};{h};{l};{c};0"


class TestDetection(unittest.TestCase):
    def test_reconnait_mt5(self):
        self.assertTrue(est_export_mt5(ENTETE_MT5))

    def test_reconnait_histdata(self):
        self.assertTrue(est_histdata("20240102 000000;1.104370;1.104510;1.104110;1.104390;0"))

    def test_ne_confond_pas_les_formats(self):
        self.assertFalse(est_histdata(ENTETE_MT5))
        self.assertFalse(est_export_mt5("20240102 000000;1.1;1.2;1.0;1.15;0"))
        self.assertFalse(est_histdata("ts,open,high,low,close,volume"))


class TestLecture(unittest.TestCase):
    def test_mt5_lignes_reelles(self):
        chemin = ecrire([
            mt5("2024.01.02", "00:00:00", 1.10437, 1.10451, 1.10411, 1.10439, 375),
            mt5("2024.01.02", "01:00:00", 1.10441, 1.10447, 1.10354, 1.10364, 2207),
        ], ENTETE_MT5)
        out = list(lignes_du_fichier(Path(chemin)))
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0][1], 1.10437)
        self.assertAlmostEqual(out[0][4], 1.10439)
        self.assertEqual(out[0][5], 375)

    def test_histdata_lignes_reelles(self):
        chemin = ecrire([
            histdata("20240102", "000000", 1.104370, 1.104510, 1.104110, 1.104390),
            histdata("20240102", "000100", 1.104390, 1.104600, 1.104300, 1.104500),
        ])
        out = list(lignes_du_fichier(Path(chemin)))
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0][1], 1.104370)
        dt = datetime.fromtimestamp(out[0][0] / 1000, tz=timezone.utc)
        self.assertEqual(dt.strftime("%Y-%m-%d %H:%M"), "2024-01-02 00:00")

    def test_ligne_cassee_ignoree_sans_tout_perdre(self):
        chemin = ecrire([
            histdata("20240102", "000000", 1.1, 1.2, 1.0, 1.15),
            "ceci n'est pas une ligne",
            histdata("20240102", "000100", 1.15, 1.25, 1.05, 1.2),
        ])
        self.assertEqual(len(list(lignes_du_fichier(Path(chemin)))), 2)


class TestDecalageHoraire(unittest.TestCase):
    def test_histdata_est_recale_en_utc(self):
        """HistData horodate en EST sans heure d'ete : 00:00 dans le fichier
        correspond a 05:00 UTC. Sans ce recalage, toute la serie est decalee."""
        chemin = ecrire([
            histdata("20240102", f"{h:02d}0000", 1.1, 1.2, 1.0, 1.15) for h in range(6)
        ])
        ad = CsvAdapter("EURUSD", "1h", chemin, tz_offset_hours=-5.0)
        premiere = ad.get_candles()[0]
        self.assertEqual(premiere.dt.strftime("%H:%M"), "05:00")
        self.assertIn("UTC-5", ad.describe())

    def test_sans_decalage_lheure_est_conservee(self):
        chemin = ecrire([
            histdata("20240102", f"{h:02d}0000", 1.1, 1.2, 1.0, 1.15) for h in range(6)
        ])
        ad = CsvAdapter("EURUSD", "1h", chemin)
        self.assertEqual(ad.get_candles()[0].dt.strftime("%H:%M"), "00:00")


class TestAgregation(unittest.TestCase):
    def _minutes(self, n, base_ts=1_700_000_000_000):
        base_ts = (base_ts // 3_600_000) * 3_600_000
        return [
            (base_ts + i * 60_000, 100 + i, 102 + i, 98 + i, 101 + i, 10)
            for i in range(n)
        ]

    def test_ohlc_de_lheure_agregee(self):
        horaires = agreger_flux(self._minutes(120), 3_600_000)
        self.assertEqual(len(horaires), 2)
        h = horaires[0]
        self.assertAlmostEqual(h.open, 100)
        self.assertAlmostEqual(h.close, 160)
        self.assertAlmostEqual(h.high, 161)
        self.assertAlmostEqual(h.low, 98)
        self.assertAlmostEqual(h.volume, 600)

    def test_ordre_des_lignes_sans_importance(self):
        """Les fichiers HistData mensuels peuvent arriver dans n'importe quel
        ordre : le resultat doit etre identique."""
        lignes = self._minutes(120)
        a = agreger_flux(lignes, 3_600_000)
        b = agreger_flux(list(reversed(lignes)), 3_600_000)
        self.assertEqual([(c.ts, c.open, c.close) for c in a],
                         [(c.ts, c.open, c.close) for c in b])

    def test_coherence_ohlc(self):
        for h in agreger_flux(self._minutes(600), 3_600_000):
            self.assertLessEqual(h.low, min(h.open, h.close))
            self.assertGreaterEqual(h.high, max(h.open, h.close))


class TestAdaptateur(unittest.TestCase):
    def _fichier_minutes(self, n, jour="20240102"):
        lignes = []
        for i in range(n):
            h, m = divmod(i, 60)
            prix = 1.1 + i * 0.0001
            lignes.append(histdata(jour, f"{h:02d}{m:02d}00",
                                   round(prix, 5), round(prix + 0.0002, 5),
                                   round(prix - 0.0002, 5), round(prix + 0.0001, 5)))
        return ecrire(lignes)

    def test_m1_lu_en_h1(self):
        ad = CsvAdapter("EURUSD", "1h", self._fichier_minutes(300))
        self.assertEqual(len(ad.get_candles()), 5)
        self.assertTrue(ad.agrege)
        self.assertIn("agrege depuis 1 min", ad.describe())

    def test_dossier_de_fichiers_mensuels_fusionne(self):
        """HistData livre un fichier par mois : le dossier doit etre lu d'un bloc."""
        dossier = Path(tempfile.mkdtemp())
        for jour in ("20240102", "20240103", "20240104"):
            lignes = [
                histdata(jour, f"{h:02d}{m:02d}00", 1.1, 1.2, 1.0, 1.15)
                for h in range(3) for m in range(0, 60)
            ]
            (dossier / f"mois_{jour}.csv").write_text("\n".join(lignes) + "\n", encoding="utf-8")
        ad = CsvAdapter("EURUSD", "1h", str(dossier))
        self.assertEqual(len(ad.get_candles()), 9)
        self.assertIn("3 fichiers", ad.describe())

    def test_dossier_vide_refuse(self):
        vide = Path(tempfile.mkdtemp())
        ad = CsvAdapter("EURUSD", "1h", str(vide))
        with self.assertRaises(MarketDataError):
            ad.get_candles()

    def test_refus_dagrandir_vers_le_bas(self):
        lignes = [histdata("20240102", f"{h:02d}0000", 1.1, 1.2, 1.0, 1.15) for h in range(10)]
        ad = CsvAdapter("EURUSD", "5m", ecrire(lignes))
        with self.assertRaises(MarketDataError):
            ad.get_candles()

    def test_format_simple_toujours_supporte(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        f.write("ts,open,high,low,close,volume\n")
        base = 1_700_000_000_000
        for i in range(10):
            f.write(f"{base + i * 3_600_000},1.1,1.2,1.0,1.15,5\n")
        f.close()
        ad = CsvAdapter("EURUSD", "1h", f.name)
        self.assertEqual(len(ad.get_candles()), 10)
        self.assertFalse(ad.agrege)


if __name__ == "__main__":
    unittest.main()


class TestPerformanceEtRobustesse(unittest.TestCase):
    """Vingt-cinq ans de M1 : la vitesse de lecture n'est pas un detail."""

    def test_calcul_de_date_identique_a_strptime(self):
        """L'optimisation ne vaut que si elle donne exactement le meme
        resultat que la methode lente qu'elle remplace."""
        import random
        from datetime import datetime, timezone
        from bot.adapters.csv_file import _parse_histdata_datetime

        rng = random.Random(0)
        for _ in range(500):
            y, mo, d = rng.randint(2000, 2026), rng.randint(1, 12), rng.randint(1, 28)
            h, mi, s = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
            dt, tt = f"{y:04d}{mo:02d}{d:02d}", f"{h:02d}{mi:02d}{s:02d}"
            attendu = int(
                datetime.strptime(f"{dt} {tt}", "%Y%m%d %H%M%S")
                .replace(tzinfo=timezone.utc).timestamp() * 1000
            )
            self.assertEqual(_parse_histdata_datetime(dt, tt), attendu)

    def test_annees_bissextiles(self):
        from datetime import datetime, timezone
        from bot.adapters.csv_file import _parse_histdata_datetime
        for date in ("20000229", "20240229", "20200229", "20250228"):
            attendu = int(
                datetime.strptime(f"{date} 120000", "%Y%m%d %H%M%S")
                .replace(tzinfo=timezone.utc).timestamp() * 1000
            )
            self.assertEqual(_parse_histdata_datetime(date, "120000"), attendu)

    def test_fichier_de_licence_ignore_sans_tout_casser(self):
        """Chaque archive HistData contient un .txt de licence a cote des
        donnees. Il ne doit pas faire echouer le chargement."""
        dossier = Path(tempfile.mkdtemp())
        lignes = [
            histdata("20240102", f"{h:02d}{m:02d}00", 1.1, 1.2, 1.0, 1.15)
            for h in range(4) for m in range(60)
        ]
        (dossier / "donnees.csv").write_text("\n".join(lignes) + "\n", encoding="utf-8")
        (dossier / "licence.txt").write_text(
            "HistData.com\nFree forex data\nTerms of use...\n", encoding="utf-8"
        )
        ad = CsvAdapter("EURUSD", "1h", str(dossier))
        self.assertEqual(len(ad.get_candles()), 4)
        self.assertIn("licence.txt", ad.fichiers_ignores)
