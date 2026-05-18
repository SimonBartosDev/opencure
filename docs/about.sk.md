---
title: O projekte OpenCure
description: Poslanie, etika a aktuálny stav platformy OpenCure.
---

<div style="position:fixed;top:14px;right:14px;z-index:9999;display:flex;font:700 13px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;border:1px solid #d0d7de;border-radius:8px;overflow:hidden;box-shadow:0 1px 5px rgba(0,0,0,.2)">
<a href="about.html" style="padding:7px 14px;text-decoration:none;background:#fff;color:#57606a">EN</a>
<a href="about.sk.html" style="padding:7px 14px;text-decoration:none;background:#2563eb;color:#fff" aria-current="page">SK</a>
</div>

# O projekte OpenCure

**[← Živý dashboard](index.html)** · [Ako to funguje (architektúra)](architecture.sk.html) · [Metodická štúdia](https://github.com/SimonBartosDev/opencure/blob/main/docs/methods_paper_draft.md) · [GitHub](https://github.com/SimonBartosDev/opencure)

> Toto je slovenský preklad. Zdrojové výpočty, kód a dáta sú v angličtine;
> v prípade rozporu je rozhodujúca [anglická verzia](about.html).

## Poslanie

OpenCure je otvorená platforma s natrvalo ukotveným poslaním, určená na
preúčelovanie liečiv (drug repurposing). Hodnotíme existujúce liečivá
schválené úradom FDA a liečivá v klinickom vývoji oproti zanedbávaným
tropickým chorobám, zriedkavým genetickým ochoreniam a ďalším
nedostatočne pokrytým indikáciám — a vytvárame predikcie, ktoré dokáže
experimentálne laboratórium overiť.

Platforma existuje preto, aby **zachraňovala životy zmenšovaním
priepasti medzi výpočtovou predikciou a laboratórnym testom**. Každá
predikcia prichádza s kalibrovanou neistotou, adverzálnou kritikou
a jednostranovým prehľadom, ktorý dokáže vedúci výskumník posúdiť za
menej než 10 minút.

Sme **nezisková iniciatíva s otvoreným zdrojovým kódom**. Celý kód je
pod licenciou Apache 2.0. Všetky natrénované modely sú uložené
v repozitári Zenodo s obsahovým odtlačkom (hash) a DOI. Všetky predikcie
sú dostupné na verejnej webovej stránke. Nikdy neprejdeme na komerčný
model, nikdy neuzamkneme dáta za API kľúč a nikdy nebudeme predávať
predikcie farmaceutickým firmám. Platforma je postavená tak, aby ju
doktorand v Nairobi alebo São Paule dokázal citovať, auditovať
a rozširovať rovnako ľahko ako výskumná skupina na MIT.

## Čo prináša verzia v7

Trinásť ortogonálnych hodnotiacich pilierov spojených do troch
ortogonálnych skupín (znalostný graf, štruktúra/fenotyp, sieť) plus
šesť samostatných signálov podľa triedy choroby. Každý kandidát
z popredných priečok prichádza s:

- **Kalibrovanou neistotou.** Konformný interval s 90 % pokrytím
  a binárna predikčná množina (`{0}`, `{1}` alebo `{0,1}`).
- **Adverzálnou kritikou.** Sedem režimov zlyhania sa kontroluje
  automaticky pri každej predikcii; voliteľný lokálny jazykový model
  prerozpráva deterministickú kritiku do textu.
- **Prehľadom pre experimentálne laboratórium.** Jednostranové zhrnutie
  vrátane navrhovaného testu (assay), koncentračného rozsahu,
  mechanistickej hypotézy a upozornení.
- **Príznakmi selektivity a esenciálnosti.** Skóre selektivity
  z databázy ChEMBL, príznak pan-esenciálnosti z DepMap a skóre dôvery
  v mechanizmus z hustoty génových asociácií OpenTargets.

Úplný opis architektúry nájdete na stránke
[Ako to funguje](architecture.sk.html).

## Aktuálny stav

- **Architektúra:** v7 — 13 aktívnych pilierov, kalibrovaná neistota,
  ensemble hlavy podľa triedy choroby, fenotypová podobnosť na základe
  obrazu, vrstvy selektivity/esenciálnosti/neistoty mechanizmu,
  adverzálny red-team agent a generátor prehľadov pre laboratóriá.
- **Skrínované choroby:** 93 (22 zanedbávaných tropických chorôb,
  19 zriedkavých chorôb, 18 onkologických, 9 kardiovaskulárnych
  a metabolických, 6 autoimunitných, 5 respiračných,
  5 neuropsychiatrických, 5 neurodegeneratívnych a 4 ďalšie
  nedostatočne pokryté).
- **Pokrytie testami:** vyše 357 regresných testov v 13 testovacích
  súboroch.
- **Reprodukovateľnosť:** každý výsledkový JSON nesie hash dátového
  manifestu a verziu pipeline, ktorá ho vytvorila; každý kontrolný bod
  modelu má obsahový odtlačok a je nahraný do Zenodo.

## Vedúce choroby (prioritné oslovenie)

Pre štyri choroby sme napísali podrobné partnerské prehľady
s mechanistickými opismi, navrhovanými testami a afiláciami cieľových
laboratórií:

- **Schistosomiáza** — DNDi, SCI Foundation, KEMRI, Imperial-Wellcome.
- **Chagasova choroba** — DNDi, Mundo Sano.
- **Kosáčikovitá anémia** — konzorcium CureSCi, Doris Duke Foundation.
- **Niemannova-Pickova choroba** — Ara Parseghian Medical Research
  Foundation, NPUK.

Jednotlivé prehľady pre 40 zanedbávaných tropických a zriedkavých
chorôb nájdete v súbore
[`lab_outreach_briefs.md`](https://github.com/SimonBartosDev/opencure/blob/main/docs/lab_outreach_briefs.md).

## Tím a kontakt

OpenCure je v súčasnosti projekt jedného vývojára s výslovným cieľom
zostať s ukotveným poslaním aj počas akéhokoľvek budúceho rastu.
Štruktúra riadenia bráni prevodu platformy na ziskový subjekt bez
hlasovania komunity.

V prípade záujmu o partnerstvo s experimentálnym laboratóriom:
`imon.bartos@gmail.com`.

## Ako citovať

Po zverejnení verzie 1 metodickej štúdie na serveri bioRxiv bude
odporúčaná citácia prepojená z domovskej stránky a vložená do každého
prehľadu pre konkrétnu chorobu.

## Licencia a etika

Kód: Apache 2.0. Dátové vklady: CC-BY 4.0 tam, kde to dovoľuje pôvodná
licencia. Zdroje trénovacích dát si zachovávajú svoje pôvodné licencie.

OpenCure poskytuje predikcie na posúdenie lekárom a výskumníkom — nejde
o odporúčania priamo pre pacientov. Netvrdíme, že niektorá konkrétna
predikcia bude prospešná konkrétnemu pacientovi. Farmakogenomické
príznaky môžu byť samy osebe skreslené zastúpením populácií
v zdrojových databázach; toto je zdokumentované pri každej predikcii.
