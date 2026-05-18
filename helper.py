import json
import os

from poke_env.battle import MoveCategory

class StateHelper:
    def __init__(self, pokedex_path="data/gen4pokedex.json", moves_path="data/gen4moves.json"):
        self.pokedex = self._load_json(pokedex_path)
        
        # Mapping for Pokemon species IDs (using the keys from your JSON)
        self.pokemon_to_id = {name: i for i, name in enumerate(sorted(self.pokedex.keys()))}
        self.unknown_pokemon_id = len(self.pokemon_to_id)

        # 18 Standard Types
        self.type_list = [
            "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE", 
            "FIGHTING", "POISON", "GROUND", "FLYING", "PSYCHIC", 
            "BUG", "ROCK", "GHOST", "DRAGON", "STEEL", "DARK", "FAIRY"
        ]
        self.type_to_idx = {t: i for i, t in enumerate(self.type_list)}

    def _load_json(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_name(self, attribute):
        """
        Helper to safely get a string name from an Enum or a string.
        """
        if attribute is None:
            return None
        # If it's a poke-env Enum (like Type.FIRE), it has a .name property
        if hasattr(attribute, "name"):
            return attribute.name
        # Fallback to string representation
        return str(attribute)

    def get_pokemon_id(self, species):
        """Returns the unique integer ID for a species."""
        name = self._get_name(species)
        if name is None:
            return self.unknown_pokemon_id
            
        clean_name = name.lower().replace(" ", "").replace("-", "")
        return self.pokemon_to_id.get(clean_name, self.unknown_pokemon_id)

    def build_state(self, battle):
        """
        Builds the complete state
        """
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # 1. Basic Battle Context
        faster = 1 if active.base_stats["spe"] > opponent.base_stats["spe"] else 0
        
        my_hp = 1 if active.current_hp_fraction < 0.5 else 2
        
        # 2. Move Information (The "Knowledge" for the Agent)
        # We create a fixed list of 4 move effectiveness values
        move_info = [0, 0, 0, 0] # Default: No move / Status move
        
        # Mapping current moves to their effectiveness
        current_moves = list(active.moves.values())
        for i, move in enumerate(current_moves):
            if i >= 4: break # Safety first
            
            if move.category == MoveCategory.STATUS:
                move_info[i] = 1 # Status move
            else:
                eff = opponent.damage_multiplier(move)
                if eff > 1: move_info[i] = 3      # Super effective
                elif eff == 1: move_info[i] = 2    # Neutral
                else: move_info[i] = 0             # Resisted / Immune

        return (faster, my_hp, *move_info)