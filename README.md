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
# Deprecated, will be removed in further versions
s, r, t, i = env.init_state(0)

# To take action in leduc poker you need:
# your action -> int where 0 = fold, 1 = call, 2 = raise
# player index -> for example 0
action = player.predict(s)
env.step(action, 0)

# Because of a two player env you need to wait until the other 
# Player has done a step aswell. Then geht new state:
# Pass your player index to function:
s, a, r, s2, t, i = env.get_new_state(0)

# If game terminates:
env.reset()
```

## Agent
The agent is the acting module of the program, in other the words:
the artifical intelligence, which should learn the optimal
strategy. It's represented by two neural networks:

1. Best Response Network
2. Average Response Network

The best response network learns with neural fitted q iteration
algorithm the best response to the opponents behaviour.

The average response network learns with simple supervised 
techniques the average strategy of itself.

The code can be found under `agent/agent.py`.

### Usage
```python
import agent.agent as agent

# Because it's ficticious self play, we need at least two
# instances of the agent and call it: player
# sess = tensorflow.Session()
# state_dim, action_dim = env.dim_shape
# learning_rate can be changed in config.ini
player1 = agent.Agent(sess, state_dim, action_dim, learning_rate)
player2 = # ...

# Act with best response network:
action = player1.best_response_model.predict(s)
env.set(action)

# Get states and remember them for replay memory
s, a, r, s2, t, i = env.get_new_state(player1_index)
player1.remember_opponent_behaviour(s, a, r, s2, t)

# When you sampled enough data, you can update the network
# It will update the Q-Function which is represented by 
# the network with supervised learning methods (because of
# replay memory).
player1.update_best_response_network()

# TODO: average response network

```


