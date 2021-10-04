from .__version__ import __version__
from safety_road_mapping.safety import SafetyMap
from safety_road_mapping.safety import generate_base_map

__pdoc__ = {
    'safety_road_mapping.safety.SafetyMap._treat_accidents_data': True,
    'safety_road_mapping.safety.SafetyMap._add_route_to_map': True,
    'safety_road_mapping.safety.SafetyMap._gen_coordinates_df': True,
    'safety_road_mapping.safety.SafetyMap._gen_sections': True,
    'safety_road_mapping.safety.SafetyMap._normalize_string': True,
    'safety_road_mapping.safety.SafetyMap._classes_accidents': True,
    'safety_road_mapping.safety.SafetyMap._filter_accident_data': True,
    'safety_road_mapping.safety.SafetyMap._days_from_accident': True,
    'safety_road_mapping.safety.SafetyMap._haversine': True,
    'safety_road_mapping.safety.SafetyMap._rank_subsections': True,
    'safety_road_mapping.safety.SafetyMap._getcolor': True,
    'safety_road_mapping.safety.SafetyMap._plot_route_score': True,
    'safety_road_mapping.safety.SafetyMap._calculate_final_score': True,
    'safety_road_mapping.safety.SafetyMap._plot_final_score': True,
    'safety_road_mapping.safety.SafetyMap._calculate_score_weight': True,
}
