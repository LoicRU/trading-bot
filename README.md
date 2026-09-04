# Bot de trading sur portefeuille fictif - V1

Simulateur de trading complet : portefeuille virtuel, stratégie qui décide
**de trader ou de ne pas trader**, système de récompense/malus, backtest
protégé contre le sur-apprentissage, et rapport HTML généré chaque soir.

**Aucun argent réel n'est engagé, et aucun ne peut l'être** : il n'existe
nulle part dans ce code de clé API de trading. L'adaptateur Binance
n'utilise que des points d'accès publics, en lecture seule.

---

## Démarrage

Aucune installation, aucune dépendance : Python 3.9+ et la bibliothèque
standard suffisent.

```bash
python3 -m bot.cli check           # vérifie la config et l'accès aux données
python3 -m bot.cli backtest        # blocs entraînement + validation
python3 -m bot.cli daily           # exécution du jour + rapport du soir
python3 -m bot.cli tick            # passage horaire : décide et journalise
python3 -m bot.cli forward-status  # état du forward test
```

Les rapports arrivent dans `reports/`, la base dans `data/bot.sqlite`.

Quatre sources de données sont disponibles, au choix dans `config.json` :

| `adapter` | Symbole | Usage |
|---|---|---|
| `binance` | `BTCUSDT` | par défaut en local — **refusé depuis les IP américaines (451)** |
| `coinbase` | `BTC-USD` | fonctionne partout, y compris depuis GitHub Actions |
| `csv` | libre | un historique téléchargé à la main |
| `synthetic` | libre | marché fabriqué, hors-ligne, pour tester la chaîne |

---

## Ce que fait le bot, concrètement

À la clôture de chaque bougie, il prend une décision parmi trois :
**acheter**, **vendre**, ou **s'abstenir** — et il écrit toujours pourquoi.
L'exécution a lieu à l'ouverture de la bougie **suivante** : il ne peut
jamais acheter au prix qui vient de déclencher son propre signal.

La stratégie livrée est volontairement simple (croisement d'EMA filtré par
l'ATR). Elle n'est pas là pour gagner de l'argent, elle est là pour que
l'infrastructure autour d'elle soit vérifiable. C'est cette infrastructure
qui a de la valeur ; la stratégie se remplace en éditant un seul fichier.

---

## Le système de récompense / malus

Deux sources de score.

**Les trades fermés** sont notés en *R-multiple* : le résultat divisé par
le risque pris à l'entrée. Toucher le stop vaut −1. Gagner trois fois son
risque vaut +3. Sans cette normalisation, le bot apprendrait simplement que
« grosse position = gros score » et augmenterait la taille jusqu'à exploser.

**Les abstentions** sont notées elles aussi, après coup :

| Situation pendant l'horizon d'évaluation | Note |
|---|---|
| Le marché est monté franchement, le bot n'était pas là | **malus** (coût d'opportunité) |
| Le marché a chuté, le bot n'était pas exposé | **bonus** (perte évitée) |
| Le marché n'a rien fait | petit **bonus** de patience |

C'est ce qui évite le piège central : si ne rien faire rapportait 0 sans
jamais rien coûter, l'inaction totale serait la stratégie mathématiquement
optimale et le bot se figerait.

Deux garde-fous dans ce calcul :

- **Les fenêtres d'évaluation ne se chevauchent pas.** Une abstention n'est
  notée qu'une fois tous les `horizon_bars`, sinon la même hausse ratée
  serait comptée 24 fois et écraserait complètement le score des trades.
- **L'évaluation regarde le futur, la décision jamais.** Noter une décision
  déjà prise avec ce qui s'est passé ensuite est légitime ; s'en servir pour
  décider ne le serait pas. Le module de score n'est donc appelé que par le
  moteur, avec un délai, et jamais par la stratégie.

---

## Discipline de validation

L'historique est coupé **chronologiquement** (jamais aléatoirement — mélanger
les dates revient à laisser le bot voir le futur) en trois blocs :

| Bloc | Part | Usage |
|---|---|---|
| `train` | 60 % | développement, essais, réglages |
| `validation` | 20 % | vérification pendant le développement |
| `holdout` (réserve) | 20 % | **une seule fois, tout à la fin** |

Le bloc de réserve est protégé par un verrou sur disque (`.holdout_used.json`).
Il faut `--use-holdout` pour y toucher, et une fois consommé il est marqué
comme tel.

La raison : si tu testes la réserve, ajustes un paramètre, puis la retestes,
elle ne vaut plus rien — tu viens de l'optimiser elle aussi, sans t'en rendre
compte. Le verrou n'est pas là pour te bloquer, il est là pour que ce soit un
geste conscient.

```bash
python3 -m bot.cli holdout-status
python3 -m bot.cli backtest --use-holdout   # quand la stratégie est figée
```

---

## Bibliothèque exploratoire de signaux (`bot/signals_explo.py`)

En plus de la liste figée de 5 signaux de prix + 2 signaux de financement
(`bot/signals.py`, testée méthodiquement dans ce projet), une bibliothèque
**exploratoire** de 40 signaux supplémentaires — construits à partir de
recherches sur TA-Lib, pandas-ta et la taxonomie des 101 alphas WorldQuant —
est disponible pour un dépistage plus large. Le catalogue complet (105
candidats recensés, 40 codés, 65 documentés pour un futur lot) est dans
[`catalogue_signaux.md`](catalogue_signaux.md).

Le protocole reste **en deux temps**, pour la même raison que le découpage
train/validation/holdout ci-dessus : tester 40 signaux d'un coup sans
correction ferait ressortir des faux positifs par pur hasard.

```bash
# Étape 1 — dépistage sur le bloc train uniquement (validation intacte)
python3 -m bot.cli explore

# Étape 2 — confirmation d'un candidat du top, UNE SEULE FOIS, sur validation
python3 -m bot.cli edge --signal <cle> --explo --segment validation
```

`explore` calcule lui-même le seuil de Bonferroni corrigé pour l'ensemble du
lot (~0.0003 avec 40 signaux × 5 horizons), classe les candidats, et propose
un top 5 à reconfirmer. La confirmation utilise le seuil corrigé de la
taille du top retenu, pas des 40 candidats de départ — c'est le dépistage qui
absorbe le coût statistique d'avoir cherché large.

---

## Gestion du risque

- **Taille calculée depuis le risque, pas depuis un montant fixe** : on décide
  d'abord ce qu'on accepte de perdre si le stop part, la taille en découle.
- Stop et objectif en multiples d'ATR : ils s'adaptent à la volatilité.
- **Plafond de perte journalière** : au-delà, plus aucun trade de la journée,
  et c'est écrit dans le rapport.
- Pas de levier, pas de vente à découvert, une position à la fois.
- **Coupe-circuit manuel** : crée un fichier nommé `KILL_SWITCH` à la racine
  et le bot refuse de démarrer.

---

## Le journal de forward testing

Le bot rejoue tout son historique à chaque exécution. C'est excellent pour la
fiabilité, mais ça pose un problème de crédibilité : si tu le lances une fois
par soir, il *recalcule* les décisions de la journée, et rien ne prouve qu'il
les aurait prises au bon moment. C'est un backtest déguisé en suivi temps réel.

`forward_log.jsonl` résout ça. À chaque passage horaire, seules les décisions
**nouvelles** y sont ajoutées, horodatées à l'heure réelle de l'enregistrement,
et jamais modifiées ensuite. Le fichier est versionné par git : l'historique
des commits date chaque ligne indépendamment de ce que le bot pourrait
raconter après coup.

C'est ce qui transforme « j'ai fait tourner mon bot six mois » en un forward
test vérifiable — l'étape 3 de la séquence avant l'argent réel.

À la création, le journal ne recopie pas l'historique : il pose seulement un
point de départ. Tout ce qui précède ne compte pas.

**Contrôle d'intégrité** : à chaque passage, les décisions déjà journalisées
sont recalculées et comparées. Un désaccord — paramètres modifiés, données
révisées par l'exchange, ou bug — est enregistré comme une ligne `divergence`
et signalé. Un forward test dont les paramètres ont bougé en cours de route
n'en est plus un, et il vaut mieux le savoir.

```bash
python3 -m bot.cli forward-status
```

---

## Mise en ligne gratuite (GitHub Actions)

Le workflow `.github/workflows/bot.yml` est prêt. Il fait un passage toutes
les heures et produit le rapport à 20h05 UTC (22h05 à Paris).

1. Crée un dépôt GitHub et pousse ce dossier.
2. **Choisis un dépôt public.** Les minutes Actions y sont illimitées, alors
   qu'un dépôt privé n'en a que 2 000 par mois — à raison d'un passage par
   heure, tu serais entre 700 et 1 400, sans marge. La V1 ne contient aucun
   secret, donc public ne t'expose à rien. Le jour où une clé API entre dans
   le projet : dépôt privé, et la clé dans les secrets GitHub, jamais dans le
   code.
3. Dans Settings → Actions → General, autorise `Read and write permissions`
   (le workflow commite le journal).
4. Onglet Actions → `Run workflow` pour le premier lancement manuel.

Le workflow utilise `config.ci.json`, identique à `config.json` **sauf**
l'adaptateur : Coinbase au lieu de Binance, parce que les runners GitHub sont
aux États-Unis et que Binance y répond 451. Un test vérifie que les deux
fichiers gardent la même stratégie — si tu modifies un paramètre, modifie-le
dans les deux, sinon ton forward test ne teste pas ce que tu as backtesté.

**Récupérer le rapport**, deux options :

- **Artefact** : téléchargeable depuis la page de l'exécution. Rien à faire.
- **Netlify** : crée un site vide, puis ajoute deux secrets dans
  Settings → Secrets and variables → Actions — `NETLIFY_AUTH_TOKEN` et
  `NETLIFY_SITE_ID`. L'étape se déclenche toute seule ; sans ces secrets elle
  est simplement sautée. Tu obtiens une URL lisible depuis ton téléphone.
  Note qu'un site Netlify est public par défaut : la protection par mot de
  passe dépend de ton offre.

---

## Exécution quotidienne en local

```bash
python3 run_daily.py
```

Sur cron, tous les soirs à 22h05 (clôture de la session de New York) :

```cron
5 22 * * *  cd /chemin/vers/trading_bot && /usr/bin/python3 run_daily.py
```

Chaque exécution rejoue tout l'historique depuis le début. C'est volontaire :
la simulation est déterministe, donc rejouer redonne exactement le même état
qu'un suivi continu — aucun risque d'état corrompu après une coupure, une
mise à jour ou un redémarrage. Les journaux vont dans `logs/daily.log`.

---

## Le rapport du soir

`reports/rapport_AAAA-MM-JJ.html`, autonome (aucun CDN, aucune dépendance),
lisible en thème clair comme sombre :

- capital, variation du jour, score du jour et cumulé, drawdown maximum ;
- courbe d'équité comparée à « acheter et conserver », avec survol ;
- tous les trades clôturés, avec leur résultat en R et le motif de sortie ;
- **toutes les décisions de ne pas trader, avec leur motif** — la section que
  les rapports de trading oublient toujours, et souvent la plus instructive ;
- les métriques cumulées.

---

## Configuration

Tout est dans `config.json`. Les réglages qui changent le plus le comportement :

| Clé | Effet |
|---|---|
| `market.adapter` | `binance`, `csv` ou `synthetic` |
| `market.symbol` / `timeframe` | paire et unité de temps |
| `portfolio.fee_rate` | frais par côté (0,001 = 0,1 %) |
| `portfolio.slippage_bps` | dégradation du prix d'exécution, toujours défavorable |
| `strategy.min_atr_pct` | sous ce seuil, marché jugé trop plat : abstention |
| `risk.risk_per_trade_pct` | fraction du capital risquée par trade |
| `risk.max_daily_loss_pct` | seuil du coupe-circuit journalier |
| `scoring.horizon_bars` | délai d'évaluation d'une abstention |

Les valeurs sont validées au démarrage : une config incohérente fait échouer
le lancement plutôt que de produire un backtest faux.

---

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

148 tests. Le plus important est `test_aucune_fuite_du_futur` : il rejoue deux
marchés identiques jusqu'à une bougie donnée puis radicalement différents
ensuite, et vérifie que **toutes les décisions passées sont rigoureusement
identiques**. Un bot qui échoue à ce test peut afficher n'importe quelle
performance, elle ne veut rien dire.

Les autres vérifient notamment qu'un aller-retour au même prix **perd** de
l'argent (frais + slippage), que la comptabilité du portefeuille tient à
chaque barre, que le piège du « ne rien faire » est bien évité, et que le
journal de forward testing ne peut jamais être réécrit après coup.

---

## Ce qui n'est pas dans la V1

Volontairement : pas d'apprentissage automatique ni de renforcement, pas de
multi-actifs simultanés, pas d'optimisation automatique des paramètres (le
meilleur moyen de sur-apprendre), pas de vente à découvert, pas de levier,
et **aucun accès à de l'argent réel**.

---

## Pour brancher un autre marché

Écris une classe qui hérite de `MarketAdapter` (trois méthodes), ajoute-la au
dictionnaire de `bot/adapters/__init__.py`. Rien d'autre ne bouge : le moteur,
le portefeuille, le scoring et le rapport sont indépendants du marché. C'est
ce qui permettra de passer au forex (OANDA) ou aux actions (Alpaca) sans
refaire le projet.

---

## Avant même de penser à de l'argent réel

1. Backtest sur `train`, mise au point de la stratégie.
2. Vérification sur `validation`.
3. **Forward testing en temps réel pendant 3 à 6 mois minimum** — l'étape que
   tout le monde saute, et la seule qui teste vraiment la stratégie sur des
   données qui n'existaient pas quand elle a été écrite.
4. Bloc de réserve, une seule fois.
5. Compte démo du broker (Binance Spot Testnet, OANDA practice), pour valider
   l'exécution réelle avec le même code.
6. Et seulement ensuite, un montant que tu es prêt à perdre entièrement.

Juge sur le **drawdown maximum** autant que sur le gain : un bot qui finit à
+15 % après être passé par −40 % est un bot que tu aurais coupé en panique
avant la fin. Et souviens-toi qu'il faut plusieurs centaines de trades avant
qu'un résultat commence à signifier quelque chose — vingt trades gagnants,
c'est du bruit.

Ce logiciel est un outil d'apprentissage. Il ne constitue pas un conseil en
investissement, et aucun résultat en simulation ne garantit quoi que ce soit
en réel.
