import json
import os
import zlib
import numpy as np

from poke_env import to_id_str

from poke_env.battle import Effect

from poke_env.battle import Effect
from poke_env.data import GenData

VOLATILE_EFFECTS_LIST = [
    Effect.SUBSTITUTE,      # Delegator
    Effect.CONFUSION,       # Verwirrung
    Effect.LEECH_SEED,      # Egelsamen
    Effect.ATTRACT,         # Anziehung
    Effect.CURSE,           # Fluch (Geist-Variante)
    Effect.NIGHTMARE,       # Nachtmar
    Effect.YAWN,            # Gähner
    Effect.PERISH3,         # Abgesang (Start-Status in poke-env)
    Effect.DESTINY_BOND,    # Abgangsbund
    Effect.ENCORE,          # Zugabe
    Effect.TAUNT,           # Verhöhner
    Effect.TORMENT,         # Folterknecht
    Effect.EMBARGO,         # Embargo
    Effect.HEAL_BLOCK,      # Heilverbot
    Effect.MAGNET_RISE,     # Magnetflug
    Effect.AQUA_RING,       # Aquaring
    Effect.INGRAIN,         # Verwurzelung
    Effect.POWER_TRICK,     # Krafttrick
    Effect.GASTRO_ACID,     # Magensaft
    Effect.MIRACLE_EYE,     # Wunderauge
    Effect.FORESIGHT,       # Gesichte
    Effect.MIND_READER,     # Willensleser
    Effect.LOCK_ON,         # Zielschuss
    Effect.FOCUS_ENERGY,    # Energiefokus
    Effect.CHARGE,          # Ladevorgang
    Effect.DEFENSE_CURL,    # Einigler
    Effect.MINIMIZE,        # Komprimator
    Effect.BIDE,            # Geduld
    Effect.RAGE,            # Rage
    Effect.BIND,            # Klammergriff (Trapping)
    Effect.CLAMP,           # Schnapper (Trapping)
    Effect.FIRE_SPIN,       # Feuerwirbel (Trapping)
    Effect.MAGMA_STORM,     # Lavasturm (Trapping)
    Effect.WHIRLPOOL,       # Whirlpool (Trapping)
    Effect.SAND_TOMB,       # Sandgrab (Trapping)
    Effect.WRAP,            # Wickel (Trapping)
    Effect.DISABLE,         # Aussetzer (Sehr wichtig in Gen 4)
    Effect.FLASH_FIRE,      # Feuerfänger (Aktivierter Status-Effekt)
]

def one_hot(value: int, min_val: int, max_val: int) -> np.ndarray:
    length = max_val - min_val + 1
    vec = np.zeros(length, dtype=np.float32)
    idx = int(clip(value, min_val, max_val) - min_val)
    vec[idx] = 1.0
    return vec

def clip(n, minn, maxn):
    return max(min(maxn, n), minn)

def bin_pp(pp_value):
    # Wang uses floor(x^(1/3)) binning for PP (Table A.0.1)
    # Result is 0 to 3 (total 4 bins)
    return int(np.floor(pp_value ** (1/3)))

def load_abilities_from_gen_data(gen_data: GenData):
    abilities = set()
    for pokemon_data in gen_data.pokedex.values():
        if "abilities" in pokemon_data:
            for ability_name in pokemon_data["abilities"].values():
                abilities.add(to_id_str(ability_name))
    return sorted(list(abilities))

def item_to_deterministic_float(item_string: str) -> float:
    if not item_string:
        return 0.0
    
    clean_id = to_id_str(item_string)
    
    checksum = zlib.crc32(clean_id.encode("utf-8")) & 0xffffffff
    
    return checksum / 0xffffffff

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