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
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        my_id = str(active.species)

        opp_id = str(opponent.species) if opponent.species else "unknown"
        
        faster = 1 if active.base_stats["spe"] > opponent.base_stats["spe"] else 0

        my_hp = int(active.current_hp_fraction * 4)
        opp_hp = int(opponent.current_hp_fraction * 4)
        
        opp_eff = active.damage_multiplier(opponent.type_1)
        threat = 0 if opp_eff > 1 else (2 if opp_eff < 1 else 1)

        best_move_eff = 0
        for move in battle.available_moves:
            eff = opponent.damage_multiplier(move)
            if eff > best_move_eff:
                best_move_eff = eff
        
        my_best_option = 3 if best_move_eff > 1 else (2 if best_move_eff == 1 else 1)

        return (my_id, opp_id, faster, my_hp, opp_hp, threat, my_best_option)