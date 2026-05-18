---
title: OpenCure v7 — Architektúra
description: Ako funguje platforma OpenCure na preúčelovanie liečiv, od začiatku do konca.
---

<div style="position:fixed;top:14px;right:14px;z-index:9999;display:flex;font:700 13px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;border:1px solid #d0d7de;border-radius:8px;overflow:hidden;box-shadow:0 1px 5px rgba(0,0,0,.2)">
<a href="architecture.html" style="padding:7px 14px;text-decoration:none;background:#fff;color:#57606a">EN</a>
<a href="architecture.sk.html" style="padding:7px 14px;text-decoration:none;background:#2563eb;color:#fff" aria-current="page">SK</a>
</div>

# OpenCure v7 — Ako to funguje

**[← Živý dashboard](index.html)** · [O projekte a poslaní](about.sk.html) · [Metodická štúdia](https://github.com/SimonBartosDev/opencure/blob/main/docs/methods_paper_draft.md) · [GitHub](https://github.com/SimonBartosDev/opencure)

> Toto je slovenský preklad. Zdrojový kód, dáta a výpočty sú v angličtine;
> v prípade rozporu je rozhodujúca [anglická verzia](architecture.html).

OpenCure je platforma s otvoreným zdrojovým kódom a natrvalo ukotveným
poslaním, ktorá hodnotí existujúce liečivá schválené úradom FDA
a liečivá v klinickom vývoji oproti zanedbávaným tropickým chorobám,
zriedkavým genetickým ochoreniam a ďalším nedostatočne pokrytým
indikáciám. Táto stránka vysvetľuje celú architektúru — každý pilier,
každý krok fúzie a každú vrstvu poctivosti — na jednom mieste.

Hlavná zásada návrhu: **žiadnej jednej metóde sa nedôveruje samostatne,
každá predikcia prichádza s kalibrovanou neistotou a každý výstup je
adverzálne kritizovaný ešte predtým, než ho uvidí človek.**

---

## 1. Problém

Vývoj lieku trvá 10 – 15 rokov a stojí viac než 2 miliardy dolárov.
Preúčelovanie už schváleného lieku preskočí väčšinu bezpečnostného
procesu — o lieku už vieme, že je u ľudí tolerovateľný. Úzkym miestom
nie sú *nápady*, ale *dôveryhodné* nápady: ktorý z desiatok miliónov
párov liek – choroba je hodný experimentu v laboratóriu?

Odpoveďou OpenCure je ohodnotiť každý pár 13 nezávislými metódami,
spojiť ich, poctivo kalibrovať neistotu, adverzálne kritizovať každého
kandidáta, ktorý prejde, a podať laboratórnemu vedcovi jednostranový
prehľad, podľa ktorého môže konať.

---

## 2. Pipeline v skratke

```
Názov choroby
  │
  ▼  Priradenie k entitám znalostného grafu (aliasy DRKG + PrimeKG + OpenTargets)
  │
  ▼  13 hodnotiacich pilierov beží paralelne
  │
  ▼  Tvrdé filtre  (kontrola SMILES → čierna listina metabolitov → heuristika IUPAC
  │                 → fáza ChEMBL → kritický ADMET, s výnimkou pre liečivá schválené FDA)
  │
  ▼  Zoskupenie pilierov + tlmenie podľa stupňa uzla (hub-degree)
  │     Skupina KG       = Reciprocal Rank Fusion (TransE, RotatE, PrimeKG, Unified, R-GCN)
  │     Štruktúrna skup. = max(odtlačky, MoLFormer-XL, DTI, JUMP Cell Painting)
  │     Sieťová skupina  = max(blízkosť STRING, reverzia génového podpisu)
  │
  ▼  Kalibrovaný ensemble  (XGBoost + izotonická kalibrácia; smerovanie podľa triedy choroby)
  │
  ▼  Obal konformnej predikcie  (interval s 90 % pokrytím + binárna predikčná množina)
  │
  ▼  Zber dôkazov  (PubMed + ClinicalTrials.gov + FAERS + Semantic Scholar; vyrovnávacia pamäť)
  │
  ▼  Klinické poistky  (uskutočniteľnosť dávky, interakcie liečiv, farmakogenomika,
  │                     triangulácia, tkanivový kontext)
  │
  ▼  Vrstvy v7  (panel selektivity, esenciálnosť z DepMap, neistota mechanizmu)
  │
  ▼  Adverzálna red-team kritika  (sedem režimov zlyhania pre každého kandidáta)
  │
  ▼  Generovanie prehľadu pre laboratórium  (jednostranový Markdown pre každú chorobu)
  │
  ▼  Dashboard + JSON + CSV + prospektívny snímok s obsahovým odtlačkom
```

Každá fáza je **bezpečná pri zlyhaní** (fail-open): ak nejaký artefakt
chýba (model ešte nie je natrénovaný, dataset ešte nie je stiahnutý),
ten pilier prispeje prázdnym výsledkom a zvyšok pipeline pokračuje.
Platforma nikdy nefunguje na princípe „všetko alebo nič“.

---

## 3. Trinásť hodnotiacich pilierov

Každý pilier je nezávislý odhad otázky „lieči tento liek túto chorobu?“,
postavený na inom druhu dôkazu. Sú zámerne *ortogonálne* — topológia
znalostného grafu, chemická štruktúra, väzba na proteín, sieťová
biológia, genetika, transkriptomika a morfológia buniek sú rôzne okná
do tej istej otázky.

| # | Pilier | Aký signál zachytáva | Zdroj dát |
|---|--------|----------------------|-----------|
| 1 | **TransE** | Embedding znalostného grafu — translačná geometria hrán liek→choroba | DRKG (5,87 mil. hrán) |
| 2 | **RotatE** | Embedding KG, kde je vzťah rotáciou; zachytáva vzory, ktoré TransE nevie | DRKG, cez PyKEEN |
| 3 | **Unified-KG TransE** | Embedding KG na *zjednotení* DRKG + PrimeKG + OpenTargets | zjednotený graf, 14 mil. hrán |
| 4 | **PrimeKG** | Nezávislý embedding KG na harvardskom grafe presnej medicíny | PrimeKG (8,1 mil. hrán) |
| 5 | **TxGNN** | Grafová neurónová sieť navrhnutá na zero-shot preúčelovanie liečiv | Harvard TxGNN, predpočítané |
| 6 | **Molekulové odtlačky** | 2D štruktúrna podobnosť so známymi liečbami (Morgan / ECFP) | RDKit |
| 7 | **MoLFormer-XL** | Naučený chemický embedding — transformer predtrénovaný na 1,1 mld. molekúl | IBM MoLFormer-XL |
| 8 | **DeepPurpose DTI** | Predikovaná väzbová afinita liek – cieľ; v7 pridáva proteínové embeddingy ESM-2 150M | BindingDB / ESM-2 |
| 9 | **Sieťová blízkosť** | Najkratšia cesta medzi cieľmi liečiva a génmi choroby v sieti interakcií proteínov | STRING v12 (473-tis. hrán) |
| 10 | **Reverzia génového podpisu** | Obracia liek transkriptomický podpis choroby? | L1000 + OpenTargets × ChEMBL |
| 11 | **Mendelovská randomizácia** | Genetický kauzálny dôkaz — je cieľ kauzálne spojený s chorobou? | OpenTargets GraphQL |
| 12 | **R-GCN** | Heterogénna grafová neurónová sieť s hlavou DistMult | natrénované na DRKG |
| 13 | **JUMP Cell Painting** | *Fenotypová* podobnosť — vyvolá liek rovnakú zmenu morfológie bunky ako známa liečba? | konzorcium JUMP-CP |

**ADMET** (Chemprop — podobnosť liečivu a toxicita, 77 ukazovateľov)
beží spolu s piliermi, ale ako ortogonálny *násobiteľ* finálneho skóre,
nie ako pilier — toxický liek treba utlmiť, nie spriemerovať.

Niekoľko pilierov si zaslúži poznámku:

- **Prečo štyri embeddingy znalostného grafu (1 – 4)?** Každý graf je
  postavený inak a má iné slepé miesta. DRKG je široký, ale z roku
  2020; PrimeKG sa zameriava na presnú medicínu; zjednotený graf
  spája všetko. Ich fúzia cez Reciprocal Rank Fusion je odolnejšia
  než dôvera v ktorýkoľvek jeden.
- **Pilier 13, JUMP Cell Painting**, je hlavný prínos verzie v7.
  Konzorcium JUMP zverejnilo ~140-tisíc *morfologických profilov*
  zlúčenín — päťkanálové fluorescenčné snímky buniek ovplyvnených
  každou zlúčeninou, zhustené do vektora príznakov. OpenCure hodnotí
  kandidáta podľa toho, ako blízko je jeho morfologický profil
  *ťažisku* známych liečieb danej choroby. Liek, ktorý je štruktúrne
  nový, no vyvoláva rovnaký bunkový fenotyp ako známa liečba, je presne
  ten cenný signál, ktorý preúčelovanie hľadá — a je to zároveň
  najväčšie zmenšenie odstupu od uzavretých platforiem so skríningom
  na základe obrazu.

---

## 4. Zoskupenie a fúzia pilierov

Viaceré piliere zachytávajú prekrývajúce sa informácie — napríklad
štyri embeddingy znalostného grafu sú silno korelované. Ak by sme do
ensemble poslali všetkých 13 surových, korelované signály by dominovali
už len svojím počtom. Preto sa piliere pred spojením **zoskupujú**:

- **Skupina KG** — TransE, RotatE, PrimeKG, Unified a R-GCN sa spájajú
  pomocou **Reciprocal Rank Fusion** (RRF). RRF kombinuje poradia, nie
  surové skóre, takže je imúnna voči veľmi odlišným škálam skóre
  piatich embeddingov.
- **Štruktúrna skupina** — odtlačky, MoLFormer-XL, DTI a JUMP Cell
  Painting sa spájajú **maximom** na zlúčeninu (najoptimistickejší
  štruktúrny/fenotypový signál).
- **Sieťová skupina** — blízkosť STRING a reverzia génového podpisu sa
  spájajú **maximom**.
- **Nezoskupené** — TxGNN, mendelovská randomizácia a ADMET zostávajú
  samostatné; sú mechanisticky natoľko odlišné, že zoskupenie by
  stratilo informáciu.

**Tlmenie podľa stupňa uzla.** Niektoré liečivá (cimetidín,
dexametazón, vápnik, glutatión) sú v znalostnom grafe prepojené takmer
so všetkým, a tak mechanicky dosahujú vysoké skóre pri *každej* chorobe.
OpenCure aplikuje na skupiny KG a siete multiplikatívnu penalizáciu
podľa stupňa uzla liečiva, kalibrovanú voči mediánu stupňa liečiv
vo fáze ≥ 1 podľa ChEMBL. Poctivé vyúčtovanie toho, čo to opravuje
a aké skreslenie pretrváva, je v súbore
[`hub_bias_analysis.md`](https://github.com/SimonBartosDev/opencure/blob/main/docs/hub_bias_analysis.md).

---

## 5. Ensemble — a hlavy podľa triedy choroby

Zoskupené skóre vstupuje do ensemble s gradientovým boostingom
(XGBoost + izotonická kalibrácia).

> **Čestné vyhodnotenie — prečítajte si toto.** Staršia verzia OpenCure
> uvádzala tento ensemble na hodnote „AUC-ROC ≈ 0,997“ v 5-násobnej
> krížovej validácii. Toto číslo bolo **únikom dát** (data leakage):
> dominantné príznaky (`transe_rank_log`, `kg_score`, ~90 % rozhodnutia
> modelu) sa počítali zo znalostného grafu, ktorý stále obsahoval práve tie
> hrany `treats`, ktoré sa použili ako testovacie značky. Model bol
> hodnotený za zapamätanie si vlastného trénovacieho grafu.
>
> Pretrénovanie bez úniku (`scripts/train_ensemble_v7.py`) — príznaky KG
> počítané z modelu so zmazanými hranami, trénované a testované len na
> pároch, ktoré model nikdy nevidel — dáva úplne iný obraz: CV AUROC ≈
> **0,72** s ťažkými negatívami a pri čestnom *časovom* teste (skutočné
> preúčelovania liečiv po roku 2020) je ensemble **na úrovni náhody alebo
> pod ňou**. Šesť jednoduchých príznakov zachytáva, ako etablovaný je liek,
> čo je v protiklade s tým, či je dané použitie skutočne *nové*.
>
> Čestný záver: **tento ensemble nepredpovedá prospektívne preúčelovanie
> liečiv.** Ponechaný je ako jeden z viacerých vstupov do hodnotenia, nie
> ako overený klasifikátor. OpenCure nezverejňuje **žiadne číslo presnosti**
> — pozri sekciu o obmedzeniach. Izotonická kalibrácia stále robí *relatívne*
> poradie skóre monotónnym, ale číselné `skóre` sa nemá čítať ako
> pravdepodobnosť úspechu.

Verzia v7 pridáva **ensemble hlavy podľa triedy choroby**. 93 chorôb je
zoskupených do šiestich terapeutických tried — *parazitárne, vírusové,
bakteriálne, onkologické, zriedkavé-metabolické, chronické-systémové* —
podľa dominantného mechanizmu preúčelovania. Každá trieda s dostatkom
trénovacích dát dostane vlastnú logistickú hlavu nad spoločnou
reprezentáciou príznakov, pretože signál, ktorý predpovedá dobré
antihelmintikum, nie je ten istý, čo predpovedá dobrý inhibítor kinázy.
Choroba, ktorej trieda má príliš málo trénovacích pozitívov, **prejde
späť na spoločnú hlavu** — smerovanie je bezpečné pri zlyhaní, nikdy
sa neuzavrie.

---

## 6. Konformná predikcia — poctivá neistota

Kalibrovaná pravdepodobnosť hovorí, že *naprieč všetkými* predikciami
s hodnotou 0,7 je správnych približne 70 %. *Nehovorí* však, aká istá
si je platforma týmto *konkrétnym* 0,7 — v skutočnosti to môže byť 0,5.

Verzia v7 túto medzeru uzatvára **rozdeľovacou konformnou predikciou**
(split conformal prediction). Vyčlenená kalibračná množina poskytne
empirický kvantil rezíduí; každá predikcia potom prichádza s:

- **intervalom nezávislým od rozdelenia** `[ensemble_prob_lower,
  ensemble_prob_upper]`, ktorý obsahuje skutočnú značku
  s pravdepodobnosťou ≥ 90 %, a
- **binárnou predikčnou množinou**: `{1}` (s istotou pozitívne), `{0}`
  (s istotou negatívne) alebo `{0, 1}` (platforma to naozaj nevie
  rozhodnúť).

Nameraná empirická miera pokrytia je **90,1 %** voči nominálnemu cieľu
90 %. Partner z laboratória, ktorý číta `prob 0,7 [0,39 – 1,00],
množina {0,1}`, vie, že platforma hovorí „pravdepodobne, ale nie som si
istá“ — čo je pravdivá odpoveď a oveľa užitočnejšia než falošná
presnosť.

---

## 7. Klinické poistky

To, čo odlišuje OpenCure od čistého radiaceho stroja: každá popredná
predikcia je *uskutočniteľná*. Každá nesie:

- **Uskutočniteľnosť dávky** — je klinické plazmatické Cmax lieku dosť
  vysoké na zasiahnutie predikovaného cieľa podľa dát o bioaktivite
  z ChEMBL?
- **Interakcie liečiv** — najnebezpečnejšie súbežné predpisy, čerpané
  z 1,4 milióna hrán interakcií v DrugBank.
- **Farmakogenomické príznaky** — upozornenia na varianty z CPIC
  a PharmGKB (HLA, CYP, VKORC1, …).
- **Mechanistická cesta** — cesta grafom v prirodzenom jazyku,
  `Liek →[inhibuje]→ Cieľ →[spojený s]→ Choroba`, nájdená ohraničeným
  prehľadávaním do šírky vo filtrovanom znalostnom grafe.
- **Triangulácia** — zhoda naprieč štyrmi nezávislými osami (znalostný
  graf, dokovanie, úroveň rozvoja cieľa Pharos, literatúra); zhoda ≥ 3
  získava označenie „strieborný štandard“.
- **Tkanivový kontext** — modifikátor expresie z GTEx, ktorý zníži váhu
  predikcie, keď gény choroby nie sú exprimované v príslušnom tkanive.

---

## 8. Vrstvy poctivosti vo verzii v7

Témou verzie v7 je *poctivosť*. Päť vrstiev existuje práve preto, aby
zachytili vlastné režimy zlyhania platformy skôr, než zavedú človeka.

- **Sada negatívnych kontrol pre 93 chorôb.** Pre každú chorobu
  kurátorsky vybraný zoznam klinicky nepravdepodobných zlúčenín
  (`tests/data/negative_controls.yaml`). Brána priebežnej integrácie
  (CI) overuje, že sa umiestnia *pod* medián danej choroby. Ak sa
  vkradne halucinovaná predikcia, CI zlyhá.
- **Panel selektivity.** Liek, ktorý sa viaže na 50 cieľov so
  submikromolárnou afinitou, je problém toxicity, nie čistý kandidát.
  Skóre selektivity (z počtu mimocieľových väzieb v ChEMBL) tlmí
  promiskuitné liečivá.
- **Príznak esenciálnosti z DepMap.** Ak je primárny cieľ lieku
  *pan-esenciálny* — nevyhnutný na prežitie v ≥ 80 % bunkových línií
  DepMap — kandidát je označený. Pan-esenciálne ciele sú liečiteľné
  v onkológii, ale predstavujú riziko systémovej toxicity inde.
- **Skóre neistoty mechanizmu.** Pri mnohých zriedkavých chorobách je
  molekulový mechanizmus zle zmapovaný. Ku každej chorobe je pripojené
  skóre dôvery 0 – 1 (odvodené z hustoty mapovania génov choroby); pod
  hodnotou 0,4 je každá predikcia označená ako špekulatívna.
- **Adverzálny red-team agent.** Každý popredný kandidát je kritizovaný
  deterministickým adverzálnym prechodom, ktorý kontroluje sedem
  režimov zlyhania: artefakty jedného piliera, nízku selektivitu,
  pan-esenciálne ciele, skreslenie podľa uzla, nízku dôveru
  v mechanizmus, nedostatok dôkazov a históriu neúspešných klinických
  skúšok. Voliteľný lokálny jazykový model prerozpráva kritiku do
  textu.

---

## 9. Výstup — prehľady pre laboratórium

Konečným artefaktom platformy nie je rebríček, ale **jednostranový
prehľad pre experimentálne laboratórium** pre každú chorobu. Pre každého
z piatich najlepších kandidátov prehľad uvádza mechanistickú hypotézu
(podloženú citáciami), navrhovaný test prispôsobený triede choroby,
koncentračný rozsah odvodený z účinnosti primárneho cieľa, red-team
kritiku a výslovné upozornenia. Štyridsať prehľadov pre zanedbávané
tropické a zriedkavé choroby je zverejnených v adresári
[`docs/outreach/`](https://github.com/SimonBartosDev/opencure/tree/main/docs/outreach);
štyri vedúce choroby — schistosomiáza, Chagasova choroba, kosáčikovitá
anémia a Niemannova-Pickova choroba — majú podrobnú kurátorskú prípravu
s konkrétne menovanými cieľovými laboratóriami.

---

## 10. Zdroje dát

| Zdroj | Čo poskytuje |
|-------|--------------|
| **DRKG** | Drug Repurposing Knowledge Graph — 5,87 mil. hrán, primárny KG |
| **PrimeKG** | Harvardský znalostný graf presnej medicíny — 8,1 mil. hrán |
| **Open Targets 24.09** | Asociácie gén – choroba, mechanizmy liek – cieľ, klinické indikácie |
| **ChEMBL 34** | 94 717 bioaktivít liek – cieľ mapovaných na DrugBank (medián IC50/Ki) |
| **STRING v12** | 473-tisíc vysoko spoľahlivých interakcií proteín – proteín |
| **GTEx v8** | Medián expresie pre 54 tkanív × ~56-tisíc génov |
| **L1000** | Transkriptomické perturbačné podpisy |
| **JUMP Cell Painting** | ~140-tisíc morfologických profilov zlúčenín |
| **DepMap** | Esenciálnosť génov (CRISPR) naprieč viac než 1000 bunkovými líniami |
| **CPIC + PharmGKB** | Farmakogenomické anotácie |
| **HGNC** | Mapovanie identifikátorov pre viac než 41-tisíc génov |
| **MoLFormer-XL, ESM-2** | Embeddingy zo základových modelov (chémia, proteíny) |

---

## 11. Stratégia validácie

- **Vyčlenené náhodné rozdelenie** — 993 párov „treats“ z DrugBank
  vyčlenených a hodnotených oproti celej množine 10 551 zlúčenín.
- **Časovo rozdelený benchmark** — 210 párov liek – choroba schválených
  *po* roku 2020, na test zovšeobecnenia za rámec znalostného grafu
  z roku 2020.
- **Pretrénovanie s odstránenými hranami** — testovacie hrany sú
  odstránené z DRKG + PrimeKG + OpenTargets pred trénovaním čistého
  modelu, aby čísla vyhľadávania neboli nadhodnotené memorovaním.
- **Konformné pokrytie** — empirických 90,1 % voči nominálnemu cieľu
  90 %.
- **Sada negatívnych kontrol** — brána CI opísaná v časti 8.
- **Priame porovnanie** — kandidáti každej choroby preradení podľa
  každého jednopilierového základu oproti spojenému ensemble.
- **Retrospektívno-prospektívna validácia** — predikcie vytvorené
  oproti dátam spred roku 2024 sú porovnané s publikáciami z rokov
  2024 – 2025, ktoré model nikdy nevidel.
- **357 automatizovaných testov** naprieč filtrami, hodnotením,
  dôkazmi, konformnou predikciou, negatívnymi kontrolami, triedami,
  JUMP-CP, selektivitou, DepMap, red-teamom a regresnými sadami,
  spúšťaných pri každom commite cez GitHub Actions.

---

## 12. Reprodukovateľnosť

Každý výsledkový súbor nesie `data_manifest_hash` — odtlačok SHA-256
každého vstupného dátového súboru, ktorý ho vytvoril. Každý kontrolný
bod modelu má obsahový odtlačok. Predikcie sa zapisujú do nemenných,
časovo označených **prospektívnych snímkov** s registráciou DOI cez
Zenodo, takže tvrdenie vyslovené dnes možno overiť oproti budúcej
literatúre. Verzia pipeline je vyznačená na každom výstupe.

---

## 13. Poctivé obmedzenia

OpenCure je zámerne otvorený o tom, čo nedokáže:

- **Žiadne vlastnícke dáta z fenotypového skríningu.** Uzavreté
  platformy (Recursion, Insitro) trénujú na miliardách interných
  snímok buniek. OpenCure to nedokáže napodobniť a ani to netvrdí.
- **Zatiaľ žiadna prospektívna validácia v laboratóriu.** Kým partnerské
  laboratórium nepotvrdí predikciu, prospektívna predikčná sila
  platformy je nepreukázaná — retrospektívne metriky môžu merať
  presakovanie dát.
- **Znalostný graf je z roku 2020.** Štvrťročná aktualizácia oproti
  aktuálnemu ChEMBL / DrugBank / OpenTargets je úlohou pre verziu v8.
- **Neistota mechanizmu je heuristika**, nie bayesovský posterior.
- **Pilier DTI** stále používa vlastný proteínový enkodér DeepPurpose;
  embeddingy ESM-2 150M sú pripravené pre budúcu DTI hlavu postavenú
  priamo na ESM-2.

Plán, ktorý sa týmto venuje, je v súbore
[`ROADMAP.md`](https://github.com/SimonBartosDev/opencure/blob/main/ROADMAP.md).

---

## 14. Kam ďalej

- **[Živý dashboard](index.html)** — prehliadajte predikcie naprieč
  93 chorobami.
- **[O projekte a poslaní](about.sk.html)** — prečo je OpenCure
  neziskový a má ukotvené poslanie.
- **[Návrh metodickej štúdie](https://github.com/SimonBartosDev/opencure/blob/main/docs/methods_paper_draft.md)**
  — text na úrovni recenzovaného článku.
- **[Prehľady pre laboratóriá](https://github.com/SimonBartosDev/opencure/blob/main/docs/lab_outreach_briefs.md)**
  — 40 prehľadov chorôb pripravených na partnerstvo.
- **[GitHub repozitár](https://github.com/SimonBartosDev/opencure)** —
  celý kód, licencia Apache 2.0.

*OpenCure poskytuje predikcie na posúdenie lekárom a výskumníkom — nejde
o odporúčania priamo pre pacientov. Každá predikcia je výpočtová
hypotéza čakajúca na experimentálne overenie.*
