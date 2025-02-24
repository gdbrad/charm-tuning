import os 
from pydantic import BaseModel
import numpy as np
import re 
from typing import Dict,Any
from decimal import Decimal


def parse_ensemble(short_tag: str) -> Dict[str, Any]:
    # see sim_params.png #
    long_tag = {
        "a125m280": "b3.30_ms-0.057_mud-0.129_s32t64-000-0001-0400",
        "test": "b3.70_ms0.000_mud-0.022_s32t96-000"
    }
    key = long_tag[short_tag]

    pattern = (
    r"b(?P<beta>[0-9]+\.[0-9]+)"
    r"_ms(?P<ms>[0-9]+\.[0-9]{3})"
    r"_mud-(?P<mud>[0-9]+\.[0-9]{3})"
    r"_s(?P<NL>[0-9]{2})"
    r"t(?P<NT>[0-9]{2})"
    r"-(?P<P>[0-9]{3})"
)
    type_map = {
        "beta": str,
        "NT": int,
        "NL":int,
        "P":str,
        "mud": str,
        "ms": str,
    }

    match = re.match(pattern, key)
    info = (
        {
            key: type_map[key](val)
            for key, val in match.groupdict().items()
            if key in type_map
        }
        if match
        else {}
)
    return info
ens_props = parse_ensemble("test")
print(ens_props)


class Spec(BaseModel):
    # new additions 
    xml_out_path: str 
    ens_short: str
    cfg_path: str
    run_path:str