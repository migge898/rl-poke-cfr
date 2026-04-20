# About
This is a practical project for the Reinforcement Learning module at Johannes Gutenberg University Mainz.\
The goal is to build an agent that uses Counterfactual Regret Minimization and Reinforcement Learning to play [Pokémon Showdown](https://pokemonshowdown.com/).\
To test it will use a team for the Format [Gen4OU](https://www.smogon.com/dex/dp/formats/ou/).

# Prerequisites
Install [Node and NPM](https://nodejs.org/en/download).

```shell
pip install -r requirements.txt
```

Clone and run Pokémon Showdown server locally:
```shell
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```