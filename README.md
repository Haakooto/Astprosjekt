# Astprosjekt

Prosjekt i AST2000, høst 2019, UiO

Håkon Olav Torvik og Tobias Baumann

Koden vi har utviklet i løpet av prosjektet til å svare på oppgaven ligger samlet i [dette git-repoet](https://github.com/Haakooto/Astprosjekt).

## Generelt

All koden vi har skrevet i dette prosjektet har vi skrevet selv, helt uten skjelettkodene. Dette står også i toppen av hver enkeltfil. Koden er strukturert i undermapper, en for hver del. Der kode er plassert i en annen del enn hvor den er spesifisert skal dette være merket med kommentarer. Unntak på dette kan forekomme.

Å putte koden for hver del i egne mapper viste seg å bli litt rotete, men det er nå slik det har blitt, og vi tar oss ikke tid til å fikse på dette, ettersom det fungerer fint.

I flere av mappene ligger det flere filer. Resten av denne README-fila gir en kort beskrivelse for hver python-fil. Dersom en pythonfil ikke beskrives her regner vi ikke den som en del av vår besvarelse.

Vi har brukt seed 76117 i alle programmene.

## Del 1
**distributions.py**
Funksjoner for plotte sannsynlighetsfordelinger.

**engine.py**
Rakettmotorklasse

**rocket.py**
Rakettklasse

**launch.py**
Simulerer rakettoppskytning og endrer referansesystem

## Del 2
**orbits.py**
Klasse for å simulere solsystemet. Brukes i hele prosjektet

**ivp.py**
Løser en difflikning for orbits.py

**two_body_system.py**
2-legeme-system, lager light curve og radial velocity curve

**n_body_system.py**
n 2-legeme-system, lager radial velocity curve

**data_analysis.py**
2 klasser for ikke-lineær regresjon

**analyse_gathered_data.py**
Bestemmer masse, radius, tetthet, og antall planeter fra radial velocity og light curve

## Del 3
**habitable_zone.py**
Regner ut og plotter beboelig sone

## Del 4
**angular_orientation.py**
Lager bilder, bestemmer angulær orientasjon. Trenger himmelkule.npy

**navigte.py**
Finner posisjon, hastighet

## Del 5
**journey.py**
Reiser fra hjemplanet til destinasjonsplanet

## Del 6
**read_spectral.py**
Analyserer datafilene fra emnesiden, trenger disse filene første gang. Vi brukte seed 17.

**atmosphere.py**
Modellerer trykk, temperatur og tetthet i atmosfæren.

## Del 7
**landing.py**
Lander

## Del 10
**HR.py**
Plotter vår stjernes utvikling i HR-diagram. Flytter første punkt til hovedsierien