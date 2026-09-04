from __future__ import annotations


def base_config_dict(**overrides) -> dict:
    """A small, valid configuration mapping for unit tests."""
    data = {
        "world": {"min_x": 0, "max_x": 100, "min_y": 0, "max_y": 100},
        "output": {"width": 101, "height": 101, "directory": "out"},
        "coordinate_system": {"flip_y": False},
        "rendering": {"marker_sizes": {0: 0, 1: 1, 2: 2, 3: 3}},
        "classifier": {"exclude": ["*_stump_*", "*_fallen*"]},
        "trees": {
            "coniferous": ["t_picea_abies_*"],
            "deciduous": ["t_betula_pendula_*", "t_sorbus_aucuparia_*"],
        },
    }
    data.update(overrides)
    return data


SYNTHETIC_LAYER = """\
$grp PolylineShapeEntity {
 {
  coords 1000 50 2000
  Points {
   ShapePoint "{6A44E11F74BA84C2}" {
    Position 0 -0 0
    Data {
     ForestGeneratorPointData "{000000002C57B0BA}" {
     }
    }
   }
  }
  LineColor 0.502 1 0 1
  {
   ForestGeneratorEntity : "{EAF4C043114BCAAB}Prefabs/WEGenerators/Forest/JD_Forest.et" {
    Flags 0x100003 0
    coords 10 0 20
    m_iSeed 23899
    {
     $grp Tree : "{009A68DF0B31B7B7}PrefabLibrary/CainLibrary/Vegetation/Trees/t_picea_abies_2sw.et" {
      {
       coords 1 -3 2
       angles 2.488 -36.652 2.561
       scale 0.816
      }
      {
       coords -5 0 5
       angles 0 0 0
       scale 1
      }
     }
     Tree : "{B497AD7A270F78C8}PrefabLibrary/CainLibrary/Vegetation/Trees/t_betula_pendula_3sw_aut.et" {
      coords 0.5 -0.5 -0.5
      angles 3.822 136.147 -2.034
      scale 1.164
     }
     $grp SCR_IndestructibleEnvironmentalEntity : "{236D8A187B6D6656}PrefabLibrary/Rocks/GraniteStone_01_V2.et" {
      {
       coords 7 0 7
       angles 0 -66.436 0
       scale 1.008
      }
     }
    }
   }
  }
 }
 {
  coords 3000 10 4000
  Points {
  }
  LineColor 0 0 0 0
  {
   ForestGeneratorEntity : "{9A1EF0571EDEB4F4}Prefabs/WEGenerators/Forest/JD_Forest_2.et" {
    coords 0 0 0
    {
     $grp Tree : "{4838F0A12B3D2AB4}PrefabLibrary/CainLibrary/Vegetation/Trees/t_betula_pendula_0_aut.et" {
      {
       coords 1 0 1
       angles 0 0 0
       scale 1
      }
     }
    }
   }
  }
 }
}
Tree : "{AAAA}PrefabLibrary/CainLibrary/Vegetation/Trees/t_sorbus_aucuparia_1w_aut.et" {
 coords 5 1 6
 angles 0 90 0
 {
  Tree : "{BBBB}PrefabLibrary/CainLibrary/Vegetation/Trees/t_sorbus_aucuparia_2s.et" {
   coords 1 0 0
  }
 }
}
"""
