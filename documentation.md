# Documentation

## Tabular Q-Learning Agent
### General

- one fixed team of size 3
- 50% training against random player, then 50% training against max damage player
- evaluation against random player (100 matches) with 87% winrate

### Actionspace

- 7 possible actions
- 4 moves and 3 switches
- only valid actions are chosen (e.g. can't switch into fainted Pokémon or use disabled move)

### Statespace

Following is saved per state:
- species of active Pokémon
- species of opponent Pokémon
- faster: 1 if own base speed > than opponent's one, else 0
- own hp in 4 bins
- opponent hp in 4 bins
- threat
    - 0 if opponent's type very effective against our types
    - 2 if not effective
    - 1 else
- my_best_option
    - 3 if effective move against opponent
    - 2 if neutral damage against opponent
    - 1 else

### Reward
- +/- 10 for opponent Pokémon fainted/own fainted calculated after move
- +/- 20 after battle win/loss

### Hyperparameter
- epsilon: 1.0
- epsilon_min: 0.05
- epsilon_decay: 0.997
- alpha: 0.1
- alpha_min: 0.001
- alpha_decay: 0.999
- gamma: 0.95