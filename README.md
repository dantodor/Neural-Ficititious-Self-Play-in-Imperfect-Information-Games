# Neuronal ficticous self-play
>  NFSP angewendetn auf Leduc-poker von David Joos <

## Struktur
### Idee
Das Programm wird in drei Module gesplitted:
1. Agent
2. Environment
3. Controller

#### Agent
Ficticous self-play ist ein Algorithmus zum finden der optimalen
Strategie indem der Agent gegen sich selbst spielt. Dabei besteht
der Agent aus zwei neuronalen Netzen.
1. Das eine Netz dient zur Vorhersage des nächsten Zuges. Bestimmt
also die eigene Strategie.
2. Das zweite Netz dient zu Vorhersage des Gegnerzuges. Diese Aussage
wird als Parameter dem ersten neuronalen Netz übergeben.

Das erste Netz lernt mit einem Reinforcement Learning Algorithmus.
Der Agent lernt dabei aus seinen eigenen Aktionen und den daraus
resultierenden Rewards.

Das zweite Netz lernt mit einem Supervised Learning Algorithmus.
Dafür nutzt er die gespielten Züge des Gegners.

#### Environment
Stellt die Spielwelt zur Verfügung. Der Agent kann die Spielwelt
aufrufen und Parameter an diese übergeben - im Umkehrschluss 
holt sich der Agent den aktuellen Stand des Spiels ebenfalls über
das Environment.

#### Controller
Initialisiert die Agenten (Spieler), das Environment und startet
das Spiel und regelt dessen Ablauf. Enthält die main() Funktion.

## TODO
1. Agenten
2. Controller



