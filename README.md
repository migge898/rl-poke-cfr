# About

This is a practical project for the Reinforcement Learning module at Johannes Gutenberg University Mainz.\
The goal is to build an agent that uses Counterfactual Regret Minimization and Reinforcement Learning to play [Pokémon Showdown](https://pokemonshowdown.com/).\
To test it will use a team for the format [Gen4OU](https://www.smogon.com/dex/dp/formats/ou/).

# Prerequisites

Install [Node and NPM](https://nodejs.org/en/download).

## (Optional) Install PyTorch for Cuda

Go to [pytorch.org](https://pytorch.org/get-started/locally/) and select your system requirements for your cuda version (Nvidia only). Otherwise pip should install the cpu-only version as default.

## Install requirements via pip
Make sure to use a virtual environment. Then install the dependencies via the following command:
```shell
pip install -r requirements.txt
```

## Clone and run Pokémon Showdown server locally
To simulate battles Pokémon Showdown is needed. Install it via this command:
```shell
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```
# Versions
## v1.0 (current)
This version is the current state that was used for the project presentation. In `checkpoints/` you can find the various trained models. And in `tensorboard_logs/` are the graphs for the different experiments.

## v2.0 (future)
It addresses the problems of v1.0 and should test the following:
- Embedding of Pokémon and battle state
- MCTS and CFR to plan actions
- gen4ou support
- evaluating against online players