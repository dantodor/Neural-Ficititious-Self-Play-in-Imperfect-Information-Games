# Neuronal ficticous self-play
>  NFSP angewendetn auf Leduc-poker von David Joos

## Environment

It simulates a leduc poker environment. The main class is `env.py` and loads the classes `deck.py`and
`card.py`.
The whole environment is found uner `leduc/`.

### Usage
```python
import leduc.env as leduc

env = leduc.Env()

# Before you take actions:
env.reset()

# To get initial state without an action for player with index 0
s, r, t, i = env.init_state(0)

# To take action in leduc poker you need:
# your action -> array with shape: [x, y, z] where x = fold, y = call, z = raise
# player index -> for example 0
s, r, t, i = env.step(action, 0)

# If game terminates:
env.reset()
```

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



