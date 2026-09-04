# Catalogue de signaux exploratoires

Reponse a la demande : chercher largement, sur internet et sur des bibliotheques
publiques (GitHub, TA-Lib, pandas-ta, la taxonomie WorldQuant, la litterature
academique), un grand nombre de signaux de trading existants, pour les
analyser ensemble plutot qu'un par un.

**105 candidats recenses.** 40 sont codes et directement testables
(`bot/signals_explo.py`, commande `python -m bot.cli explore`). 65 sont
documentes ci-dessous avec leur source mais **pas encore codes** — un futur
lot, pas une extension silencieuse du premier.

## Pourquoi ce n'est pas juste "tester les 105 et garder ce qui marche"

Ce projet a decouvert deux fois, en le vivant, ce qui se passe quand on
teste beaucoup de candidats et qu'on garde les gagnants apparents : un faux
positif crypto ("creux") mort sur le decoupage validation, un faux positif
forex ("rsi_rebond") mort quand l'historique a ete multiplie par 10. Avec
100+ candidats, le probleme est mecaniquement pire : a seuil brut de 5 %,
tester 100 signaux x 5 horizons = 500 tests fait attendre **25
"significatifs" par pur hasard**, meme si strictement rien ici n'a de valeur.

Le protocole retenu (deja code, voir `bot/signals_explo.py` pour le detail) :

1. **Depistage** (`python -m bot.cli explore`) — les 40 signaux codes sont
   testes sur le bloc TRAIN uniquement, avec un seuil de Bonferroni tres
   strict (~0.0003 sur ce lot). Ce n'est pas un verdict, c'est un tri.
2. **Confirmation** — les 3 a 5 meilleurs du tri sont retestes UNE FOIS sur
   le bloc VALIDATION (`edge --signal <cle> --explo --segment validation`),
   avec un seuil corrige de la taille du *top* retenu, pas des 100+ candidats
   de depart. C'est cette etape qui tranche, pas la premiere.
3. Le bloc RESERVE (holdout) reste hors de tout ceci — il sert uniquement au
   verdict final de la strategie complete, une fois figee.

Les 65 candidats "pas encore codes" ci-dessous ne rentrent PAS dans l'etape 1
actuelle. Les ajouter au fichier `CANDIDATS_EXPLO` apres avoir vu les
resultats du premier lot invaliderait la correction statistique deja
appliquee — exactement l'erreur que ce protocole existe pour eviter. S'ils
sont codes un jour, ce sera comme un lot 3 pre-enregistre a part entiere,
avec sa propre correction — exactement comme la serie "financement" l'a ete
pour la serie "prix" dans `bot/signals.py`.

---

## Partie 1 — Codes et testables aujourd'hui (`bot/signals_explo.py`)

### rebond_survente

1. **`stoch_rebond`** — Stochastique lent sort de survente (%K > 20)  
   Hypothese : retour a la moyenne : le %K stochastique repasse au-dessus de 20  
   Source : TA-Lib/pandas-ta: Stochastic Oscillator

2. **`stoch_rapide_rebond`** — Stochastique rapide sort de survente (%K(5) > 20) *(correle avec : stoch_rebond)*  
   Hypothese : variante rapide du precedent : periode plus courte, plus de signaux, plus de bruit  
   Source : TA-Lib: STOCHF

3. **`williams_rebond`** — Williams %R sort de survente (> -80) *(correle avec : stoch_rebond (formule quasi identique))*  
   Hypothese : retour a la moyenne, formule equivalente au stochastique inverse  
   Source : TA-Lib: WILLR

4. **`cci_rebond`** — Commodity Channel Index sort de survente (> -100)  
   Hypothese : le prix typique s'ecarte fortement sous sa moyenne puis y revient  
   Source : TA-Lib: CCI

5. **`mfi_rebond`** — Money Flow Index sort de survente (> 20)  
   Hypothese : RSI pondere par le volume : la pression vendeuse epuise le volume, puis se retourne  
   Source : TA-Lib: MFI

6. **`rsi_court_rebond`** — RSI(9) sort de survente (> 25) *(correle avec : rsi_rebond (bot/signals.py))*  
   Hypothese : variante plus courte et plus stricte du rsi_rebond deja teste (periode 14, seuil 30)  
   Source : pratique courante (RSI court terme)

7. **`rsi_survente_profonde`** — RSI(14) sort d'une survente extreme (> 20) *(correle avec : rsi_rebond (bot/signals.py))*  
   Hypothese : hypothese distincte : seuls les exces VRAIMENT extremes (pas 30, mais 20) ont un avantage  
   Source : variante de seuil, litterature retail

8. **`stoch_kd_cross`** — %K stochastique croise au-dessus de %D *(correle avec : stoch_rebond)*  
   Hypothese : signal de retournement classique, different du simple seuil de survente  
   Source : TA-Lib/pandas-ta: Stochastic

### momentum

9. **`roc_court_zero`** — Rate of Change(12) repasse positif  
   Hypothese : momentum brut a court terme, sans lissage moyenne mobile  
   Source : TA-Lib: ROC

10. **`roc_long_zero`** — Rate of Change(25) repasse positif *(correle avec : roc_court_zero)*  
   Hypothese : meme logique que roc_court_zero mais sur un horizon plus lent  
   Source : TA-Lib: ROC

11. **`macd_hist_zero`** — Histogramme MACD repasse positif  
   Hypothese : l'ecart entre la ligne MACD et son signal change de sens : acceleration du momentum  
   Source : TA-Lib/pandas-ta: MACD

12. **`awesome_osc_zero`** — Awesome Oscillator repasse positif  
   Hypothese : SMA(5)-SMA(34) du point milieu : momentum de tendance, insensible au bruit intra-bougie  
   Source : pandas-ta: Awesome Oscillator (Bill Williams)

13. **`cloture_force_range`** — La cloture capture >70% de l'amplitude haut-bas de la bougie (force acheteuse intra-bougie)  
   Hypothese : inspire de la famille 'structure intra-journaliere' des alphas WorldQuant : une cloture proche du plus haut de la bougie trahit une pression acheteuse en fin de periode  
   Source : WorldQuant 101 Alphas (famille intraday structure, adapte en mono-actif)

### tendance

14. **`macd_ligne_zero`** — Ligne MACD repasse positive *(correle avec : macd_hist_zero, ema_cross_moyen)*  
   Hypothese : la moyenne courte repasse au-dessus de la longue en valeur absolue, pas seulement l'une vs l'autre  
   Source : TA-Lib/pandas-ta: MACD

15. **`trix_zero`** — TRIX repasse positif  
   Hypothese : taux de variation d'une triple EMA : filtre le bruit court terme mieux qu'une EMA simple  
   Source : TA-Lib: TRIX

16. **`macd_signal_cross`** — Ligne MACD croise au-dessus de son signal *(correle avec : macd_hist_zero (les deux se declenchent presque ensemble))*  
   Hypothese : la formulation la plus classique du MACD, distincte des versions 'ligne vs zero'  
   Source : TA-Lib/pandas-ta: MACD

17. **`ema_cross_court`** — EMA(5) croise au-dessus de EMA(20) *(correle avec : ema_cross (bot/signals.py))*  
   Hypothese : suivi de tendance tres reactif, variante courte du croisement d'EMA deja teste  
   Source : pratique courante

18. **`ema_cross_moyen`** — EMA(10) croise au-dessus de EMA(50) *(correle avec : ema_cross (bot/signals.py), ema_cross_court)*  
   Hypothese : suivi de tendance a horizon intermediaire  
   Source : pratique courante

19. **`ema_cross_long`** — EMA(50) croise au-dessus de EMA(200) (« golden cross ») *(correle avec : ema_cross (bot/signals.py))*  
   Hypothese : le croisement de suivi de tendance le plus etudie dans la litterature retail  
   Source : pratique courante (golden/death cross)

20. **`sma_cross`** — SMA(20) croise au-dessus de SMA(50) *(correle avec : ema_cross_moyen)*  
   Hypothese : meme logique que le croisement d'EMA, avec un lissage different (SMA non pondere)  
   Source : pratique courante

21. **`vortex_cross`** — VI+ croise au-dessus de VI-  
   Hypothese : mouvement directionnel haussier depasse le mouvement directionnel baissier  
   Source : TA-Lib/pandas-ta: Vortex Indicator

22. **`aroon_cross`** — Aroon Up croise au-dessus de Aroon Down  
   Hypothese : le plus haut recent redevient plus proche que le plus bas recent  
   Source : TA-Lib/pandas-ta: Aroon

### cassure

23. **`bollinger_cassure_20`** — Cloture franchit la bande de Bollinger haute (20, 2σ) *(correle avec : cassure (bot/signals.py))*  
   Hypothese : momentum reel : le prix sort de son enveloppe de volatilite habituelle  
   Source : TA-Lib/pandas-ta: Bollinger Bands

24. **`bollinger_cassure_10`** — Cloture franchit la bande de Bollinger haute (10, 2.5σ) *(correle avec : bollinger_cassure_20)*  
   Hypothese : variante plus etroite/rapide de bollinger_cassure_20  
   Source : TA-Lib/pandas-ta: Bollinger Bands

25. **`keltner_cassure`** — Cloture franchit le canal de Keltner haut *(correle avec : bollinger_cassure_20)*  
   Hypothese : meme idee que Bollinger, mais l'enveloppe est batie sur l'ATR plutot que l'ecart-type  
   Source : TA-Lib/pandas-ta: Keltner Channel

26. **`donchian_cassure_20`** — Cassure du plus haut des 20 dernieres bougies *(correle avec : cassure (bot/signals.py))*  
   Hypothese : variante plus courte de la cassure deja testee (48 bougies) : plus de signaux, plus de bruit  
   Source : TA-Lib: variante Donchian

27. **`donchian_cassure_96`** — Cassure du plus haut des 96 dernieres bougies *(correle avec : cassure (bot/signals.py))*  
   Hypothese : variante plus longue : moins de signaux, cassures plus significatives  
   Source : TA-Lib: variante Donchian

28. **`donchian_cassure_200`** — Cassure du plus haut des 200 dernieres bougies *(correle avec : cassure (bot/signals.py))*  
   Hypothese : horizon multi-mois : capture les cassures de range majeures uniquement  
   Source : TA-Lib: variante Donchian

### retour_moyenne

29. **`bollinger_rebond_bas`** — Cloture rentre dans la bande apres etre passee sous la bande basse *(correle avec : creux (bot/signals.py))*  
   Hypothese : exces baissier statistique : le prix revient dans son enveloppe habituelle  
   Source : TA-Lib/pandas-ta: Bollinger Bands

30. **`keltner_rebond_bas`** — Cloture rentre dans le canal apres etre passee sous le canal bas *(correle avec : bollinger_rebond_bas, creux (bot/signals.py))*  
   Hypothese : meme idee que bollinger_rebond_bas avec une enveloppe batie sur l'ATR  
   Source : TA-Lib/pandas-ta: Keltner Channel

31. **`donchian_rebond_bas`** — Cloture rentre au-dessus du plus bas des 20 dernieres bougies *(correle avec : creux (bot/signals.py))*  
   Hypothese : rebond sur un support recent, apres l'avoir teste par en dessous  
   Source : TA-Lib: variante Donchian

### volume

32. **`cmf_zero`** — Chaikin Money Flow repasse positif  
   Hypothese : pression acheteuse (cloture proche du haut, volume a l'appui) redevient dominante  
   Source : TA-Lib/pandas-ta: Chaikin Money Flow

33. **`obv_roc_zero`** — Taux de variation de l'On Balance Volume repasse positif  
   Hypothese : le volume cumule signe change de tendance avant (ou avec) le prix  
   Source : TA-Lib: OBV + ROC compose

34. **`volume_spike_haussier`** — Pic de volume (>2x la moyenne 20) sur une bougie haussiere  
   Hypothese : un volume anormal a l'appui d'une hausse trahit un interet institutionnel  
   Source : pratique courante (volume breakout)

35. **`divergence_obv_haussiere`** — Le prix fait un nouveau plus bas mais l'OBV non *(correle avec : creux (bot/signals.py))*  
   Hypothese : divergence haussiere classique : le volume cumule ne confirme pas la baisse du prix, signe d'accumulation silencieuse  
   Source : analyse technique classique (divergence prix/volume)

### composite

36. **`confirmation_tendance_momentum`** — RSI(14) traverse 50 pendant que EMA(20) > EMA(50) *(correle avec : ema_cross_moyen)*  
   Hypothese : signal compose : le momentum se retourne A L'INTERIEUR d'une tendance haussiere deja en place, plutot que contre elle  
   Source : pratique courante (filtre de tendance + momentum)

37. **`cassure_avec_volume`** — Cassure de plus haut (48 bougies) confirmee par un volume >1.5x la moyenne *(correle avec : cassure (bot/signals.py))*  
   Hypothese : filtre la cassure deja testee (bot/signals.py) par une confirmation de volume : une cassure sans volume est souvent un faux signal  
   Source : pratique courante (breakout + volume)

38. **`triple_confirmation`** — MACD haussier + %K>%D + cloture au-dessus de l'EMA(50), simultanement *(correle avec : macd_signal_cross, stoch_kd_cross)*  
   Hypothese : trois familles d'indicateurs independantes (momentum, oscillateur, tendance) qui s'accordent en meme temps : idee de vote/consensus a petite echelle  
   Source : approche 'confluence' courante en analyse technique

### volatilite

39. **`compression_expansion`** — Bande de Bollinger au plus etroit depuis 100 bougies, puis cassure de la bande haute *(correle avec : bollinger_cassure_20)*  
   Hypothese : la volatilite se contracte avant de se relacher (« squeeze ») ; la cassure qui suit une compression serait plus fiable qu'une cassure ordinaire  
   Source : pratique courante (Bollinger Band Squeeze / TTM Squeeze)

40. **`compression_expansion_atr`** — ATR relatif au plus bas depuis 100 bougies, puis bougie haussiere forte (>1.5x l'ATR) *(correle avec : compression_expansion)*  
   Hypothese : variante de compression_expansion basee sur l'ATR plutot que la largeur de Bollinger : la volatilite realisee (pas seulement l'ecart-type des clotures) se contracte puis explose  
   Source : pratique courante (volatility contraction pattern)

---

## Partie 2 — Recenses, sourcés, PAS ENCORE codés (lot futur)

Ces 65 candidats sont issus de la meme recherche (TA-Lib, pandas-ta,
awesome-quant, la taxonomie WorldQuant, la litterature academique sur les
facteurs). Ils ne sont pas dans `bot/signals_explo.py` et ne doivent pas y
etre ajoutes silencieusement — voir l'avertissement en tete de ce fichier.

### Figures de bougies (candlestick patterns) — TA-Lib `CDL*`

41. **Engulfing haussier** (`CDLENGULFING`) — une bougie haussiere avale entierement la precedente bougie baissiere.
42. **Marteau** (`CDLHAMMER`) — petite meche haute, longue meche basse en fin de baisse.
43. **Marteau inverse** (`CDLINVERTEDHAMMER`) — variante symetrique du marteau.
44. **Etoile du matin** (`CDLMORNINGSTAR`) — motif a trois bougies signalant un retournement de fond.
45. **Perce-nuage** (`CDLPIERCING`) — bougie haussiere qui referme au-dela de la moitie du corps baissier precedent.
46. **Trois soldats blancs** (`CDL3WHITESOLDIERS`) — trois bougies haussieres consecutives, cloture pres du plus haut.
47. **Harami haussier** (`CDLHARAMI`) — petite bougie contenue dans le corps de la precedente, signe d'indecision suivi d'un retournement.
48. **Doji dragon-fly** (`CDLDRAGONFLYDOJI`) — ouverture=cloture avec longue meche basse.
49. **Marubozu haussier** (`CDLMARUBOZU`) — bougie sans meche, corps plein : conviction directionnelle maximale.
50. **Bebe abandonne haussier** (`CDLABANDONEDBABY`) — gap puis doji puis gap inverse : retournement violent.
51. **Trois interieurs haussier** (`CDL3INSIDE`) — harami confirme par une troisieme bougie.
52. **Ceinture haussiere** (`CDLBELTHOLD`) — ouverture au plus bas, forte cloture haussiere.
53. **Kicking haussier** (`CDLKICKING`) — deux marubozu opposes separes par un gap.
54. **Trois methodes ascendantes** (`CDLRISETHREE`... variante) — pause de consolidation dans une tendance haussiere, puis reprise.
55. **Tweezer bottom** (motif compose, pas une fonction TA-Lib unique) — deux bas quasi identiques consecutifs, rejet d'un niveau.

### Moyennes mobiles et tendance additionnelles

56. **KAMA cross** — Kaufman Adaptive Moving Average (s'adapte a la volatilite) croise le prix. *(pandas-ta)*
57. **Hull MA cross** — moyenne mobile de Hull, reduit le lag par rapport a une EMA classique. *(pandas-ta)*
58. **T3 cross** — Triple EMA lissee (T3), moins de faux signaux qu'une EMA simple en range. *(TA-Lib)*
59. **TEMA/DEMA cross** — double/triple EMA, reduction du lag. *(TA-Lib)*
60. **Ichimoku : prix sort du nuage (Kumo) par le haut** — le systeme Ichimoku complet, tendance + support/resistance dynamique. *(pandas-ta: Ichimoku)*
61. **Ichimoku : Tenkan-sen croise au-dessus de Kijun-sen** — equivalent Ichimoku d'un croisement de moyennes rapides/lentes.
62. **Parabolic SAR : bascule haussiere** — le point SAR passe sous le prix, signal de suivi de tendance avec stop integre. *(TA-Lib: SAR)*
63. **ADX/DMI : ADX en hausse et +DI croise au-dessus de -DI** — force de tendance ET direction combinees. *(TA-Lib: ADX, PLUS_DI, MINUS_DI)*
64. **Pente de regression lineaire qui redevient positive** — LINEARREG_SLOPE change de signe. *(TA-Lib: LINEARREG_SLOPE)*
65. **Z-score du prix vs SMA(50) qui remonte au-dessus de -2** — retour a la moyenne parametrique, apparente a `creux` mais normalise par l'ecart-type plutot que l'ATR.
66. **Midpoint/Midprice cross** — variantes TA-Lib peu utilisees, a tester par completude. *(TA-Lib: MIDPOINT, MIDPRICE)*

### Volume et flux additionnels

67. **Price Volume Trend (PVT) croise zero** — cumul du volume pondere par la variation relative du prix. *(TA-Lib/pandas-ta)*
68. **Negative Volume Index (NVI) croise sa propre moyenne** — isole les jours a faible volume, reputes porter l'information "smart money". *(pandas-ta)*
69. **Positive Volume Index (PVI) croise sa propre moyenne** — symetrique du precedent, jours a fort volume.
70. **Ease of Movement croise zero** — relie l'amplitude de prix au volume necessaire pour la produire. *(pandas-ta)*
71. **Elder's Force Index croise zero** — (cloture-cloture precedente) x volume, lisse. *(pandas-ta)*
72. **Klinger Volume Oscillator croise sa ligne de signal** — oscillateur de volume long terme.
73. **Volume Profile : cassure au-dessus du point de controle (POC)** — le niveau de prix ayant concentre le plus de volume recent sert de pivot.
74. **Accumulation/Distribution Line (AD) croise zero** — proche de l'OBV, pondere par la position de cloture dans le range. *(TA-Lib: AD)*
75. **VWAP : cloture repasse au-dessus du prix moyen pondere par le volume** — reference institutionnelle classique.

### Oscillateurs additionnels

76. **Ultimate Oscillator sort de survente (>30)** — combine trois periodes pour reduire les faux signaux. *(TA-Lib: ULTOSC)*
77. **Relative Vigor Index croise son signal** — mesure la conviction de la cloture par rapport a l'ouverture. *(pandas-ta)*
78. **Fisher Transform croise zero** — transforme le prix pour accentuer les points de retournement statistiquement. *(pandas-ta)*
79. **Chande Momentum Oscillator sort de survente (>-50)** — momentum normalise entre -100 et 100. *(pandas-ta)*
80. **Coppock Curve croise zero** — concu pour detecter les creux de marche majeurs (usage historique sur indices actions). *(pandas-ta)*
81. **KST (Know Sure Thing) croise son signal** — somme ponderee de plusieurs ROC lisses. *(pandas-ta)*
82. **Detrended Price Oscillator (DPO) croise zero** — retire la composante de tendance pour isoler les cycles. *(pandas-ta: DPO)*
83. **Balance of Power croise zero** — (cloture-ouverture)/(haut-bas), proche de `cloture_force_range` deja code mais sans normalisation par fenetre. *(TA-Lib: BOP)*
84. **Center of Gravity croise son signal** — oscillateur de John Ehlers, faible lag. *(pandas-ta)*
85. **STOCHRSI sort de survente** — RSI applique a lui-meme via le stochastique, plus reactif qu'un RSI simple. *(TA-Lib: STOCHRSI)*

### Inspires de la taxonomie WorldQuant (adaptes en mono-actif — a l'origine concus pour un panier d'actifs)

86. **Correlation glissante prix/volume qui passe de negative a positive** — retour d'un accord entre volume et prix. *(famille "price-volume divergence")*
87. **VWAP reversion** — cloture repasse au-dessus du VWAP journalier apres etre passee dessous. *(famille "intraday structure")*
88. **Comblement de gap baissier** — grande bougie de gap baissier suivie d'une bougie de reprise. *(famille "volatilite/reversion")*
89. **Ratio d'expansion de range intra-bougie** — (haut-bas) de la bougie courante vs sa moyenne recente, croise a la hausse avec une cloture forte. *(famille "volatilite")*
90. **Alpha#101-like : (cloture-ouverture)/((haut-bas)+0.001)** — formule quasi-litterale de l'alpha 101 original de la publication WorldQuant, seuil a calibrer.

### Facteurs academiques (majoritairement PENSES pour du transversal — `bot/pooled.py`, pas un seul actif)

91. **Momentum 12-1 mois** — rendement sur 252 jours en excluant le dernier mois (Jegadeesh & Titman 1993), le facteur momentum le plus reproduit de la litterature. Cross-sectionnel par nature.
92. **Reversion a court terme (1 semaine)** — inverse du precedent sur un horizon tres court (Lehmann 1990, Jegadeesh 1990).
93. **Proximite du plus haut sur 52 semaines** — les actifs proches de leur plus haut annuel continueraient a surperformer (George & Hwang 2004). Cross-sectionnel.
94. **Facteur basse volatilite** — les actifs les moins volatils surperformeraient sur base ajustee du risque (anomalie "low-vol", Baker/Bradley/Wurgler). Necessite un classement transversal, pas un seul actif.
95. **Volatilite idiosyncratique** — composante de volatilite non expliquee par le marche general ; necessite un indice de reference (BTC dominance ou un indice large) non encore integre au projet.
96. **Facteur de carry (deja identifie dans la conversation)** — differentiel de taux d'interet entre devises ; necessite des donnees de taux non encore integrees. Piste serieuse deja discutee, distincte de ce lot technique.
97. **Facteur de qualite/liquidite (volume moyen relatif)** — les paires les plus liquides de l'univers surperformeraient sur cout-ajuste ; a tester via `bot/pooled.py`.
98. **Retournement du sentiment de financement extreme sur plusieurs paires simultanement** — extension transversale de `financement_bas`/`financement_haut` (deja dans `bot/signals.py`), teste paire par paire mais jamais en panier.
99. **Facteur de skewness des rendements** — les actifs a asymetrie negative recente rebondiraient davantage (lien avec le "creux" deja teste, en version cross-sectionnelle).
100. **Facteur de dispersion sectorielle** — quand la dispersion des rendements entre paires augmente, les strategies relatives (long/short) auraient plus de marge. Necessite un panier long/short, hors du perimetre "long seulement" actuel du bot.

### Combinaisons/meta-signaux (a construire APRES avoir des briques individuelles qui marchent, pas avant)

101. **Vote majoritaire sur un sous-ensemble de 5 a 7 signaux decorreles** — l'idee "ensemble" deja discutee ; n'a de sens que si au moins quelques briques individuelles ci-dessus passent la confirmation.
102. **Regime switching** — n'activer les signaux de suivi de tendance qu'en regime de tendance detecte (ADX haut) et les signaux de retour a la moyenne qu'en regime de range (ADX bas).
103. **Meta-signal calendaire** — jour de la semaine / heure de la journee comme filtre (effets calendaires documentes sur actions, jamais testes ici sur crypto/forex).
104. **Filtre de correlation au marche large** — n'entrer que si un indice de reference (BTC dominance, DXY pour le forex) confirme la meme direction.
105. **Meta-signal de largeur de marche transversale** — proportion de paires de l'univers `bot/universe.py` qui donnent un signal d'achat simultanement, comme filtre de contexte plutot que comme signal en soi.

---

## Ce que ce catalogue ne remplace pas

Un signal sur cette liste, code ou non, reste une HYPOTHESE, pas un
resultat. La quasi-totalite des candidats deja testes serieusement dans ce
projet (5 signaux de prix + 2 signaux de financement, sur 8 ans de BTC et
sur EUR/USD) n'ont pas survecu a la correction statistique. Il n'y a aucune
raison a priori que ce nouveau lot de 40 fasse mieux — la seule chose que
change ce catalogue, c'est qu'il est desormais possible de le VERIFIER
rapidement plutot que de le supposer.
