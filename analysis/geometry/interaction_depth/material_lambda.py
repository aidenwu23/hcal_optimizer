#!/usr/bin/env python3
"""Resolve nuclear interaction lengths for materials used by the HCAL study."""

from __future__ import annotations

import argparse
import ast
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set
import xml.etree.ElementTree as xml_tree

PROJECT_DIRECTORY = Path(__file__).resolve().parents[3]
ELEMENTS_XML_PATH = PROJECT_DIRECTORY / "geometries" / "definitions" / "elements.xml"
MATERIALS_XML_PATH = PROJECT_DIRECTORY / "geometries" / "definitions" / "materials.xml"

LENGTH_TO_MM = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
}

DENSITY_TO_G_CM3 = {
    "g/cm3": 1.0,
    "mg/cm3": 0.001,
}

# One material component from the XML graph. This can be either a mass fraction or a
# stoichiometric count, depending on how the material was written in the XML.
@dataclass
class MaterialPiece:
    name: str
    amount: float
    amount_kind: str


# One node in the material graph after the XML files are loaded. Primitive elements can
# carry tabulated lambda_I directly, while mixtures and compounds carry component pieces.
@dataclass
class MaterialEntry:
    name: str
    density_g_cm3: float | None = None
    lambda_I_mm: float | None = None
    atomic_mass_g_mol: float | None = None
    pieces: List[MaterialPiece] = field(default_factory=list)
    lambda_I_resolved_mm: float | None = None
    mass_resolved_g_mol: float | None = None


# Keep the loaded material graph in one container so it can be passed around the analysis
# without re-reading the XML files.
@dataclass
class MaterialLibrary:
    entries_by_name: Dict[str, MaterialEntry]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve material nuclear interaction lengths from the repo XML files."
    )
    parser.add_argument(
        "--material",
        action="append",
        default=[],
        help="Material name to resolve. Repeat for multiple materials.",
    )
    return parser.parse_args()


# Read one of the XML material definition files into memory before walking its material tree.
def _read_xml_root(xml_path: Path) -> xml_tree.Element:
    if not xml_path.exists():
        raise FileNotFoundError(f"Material definition file not found: {xml_path}")
    return xml_tree.parse(xml_path).getroot()


# The XML files use short arithmetic expressions in some composition fields, so evaluate
# those expressions with a tightly restricted parser before using them as physical weights.
def _eval_expression_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_expression_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Numeric expression contains a non-numeric constant.")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_expression_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.Add, ast.Sub, ast.Mult, ast.Div),
    ):
        left_value = _eval_expression_node(node.left)
        right_value = _eval_expression_node(node.right)
        if isinstance(node.op, ast.Add):
            return left_value + right_value
        if isinstance(node.op, ast.Sub):
            return left_value - right_value
        if isinstance(node.op, ast.Mult):
            return left_value * right_value
        return left_value / right_value
    raise ValueError("Numeric expression contains unsupported syntax.")


def _eval_number_expression(number_text: str) -> float:
    """Evaluate a simple XML numeric expression used in material fractions."""
    parsed_expression = ast.parse(number_text.strip(), mode="eval")
    return float(_eval_expression_node(parsed_expression))


# Read the tabulated density when it is present. The interaction-length mixture rule uses
# mass fractions, but keeping the density is still useful context for later checks.
def _read_density_g_cm3(material_element: xml_tree.Element) -> float | None:
    """Read the material density in g/cm3 when it is present."""
    density_element = material_element.find("D")
    if density_element is None:
        return None
    value_text = density_element.get("value")
    if value_text is None:
        return None
    unit_text = (density_element.get("unit") or "g/cm3").replace(" ", "")
    if unit_text not in DENSITY_TO_G_CM3:
        raise ValueError(f"Unsupported density unit '{unit_text}' in {material_element.get('name')}")
    return _eval_number_expression(value_text) * DENSITY_TO_G_CM3[unit_text]


# Primitive entries in elements.xml already carry a nuclear interaction length. Those
# tabulated values are the anchors for every recursive mixture calculation downstream.
def _read_lambda_I_mm(material_element: xml_tree.Element) -> float | None:
    """Read the direct nuclear interaction length in mm when it is tabulated."""
    nil_element = material_element.find("NIL")
    if nil_element is None:
        return None
    value_text = nil_element.get("value")
    if value_text is None:
        return None
    unit_text = nil_element.get("unit") or "mm"
    if unit_text not in LENGTH_TO_MM:
        raise ValueError(f"Unsupported NIL unit '{unit_text}' in {material_element.get('name')}")
    return _eval_number_expression(value_text) * LENGTH_TO_MM[unit_text]


# Stoichiometric materials need atomic masses so their integer composition can be turned
# into the mass fractions required by the interaction-length mixture rule.
def _read_atomic_mass_g_mol(element_element: xml_tree.Element) -> float:
    """Read the atomic mass needed to convert stoichiometric counts into mass fractions."""
    atom_element = element_element.find("atom")
    if atom_element is None:
        raise ValueError(f"Element {element_element.get('name')} is missing <atom>.")
    value_text = atom_element.get("value")
    if value_text is None:
        raise ValueError(f"Element {element_element.get('name')} is missing atomic mass.")
    unit_text = (atom_element.get("unit") or "g/mol").replace(" ", "")
    if unit_text != "g/mol":
        raise ValueError(f"Unsupported atomic-mass unit '{unit_text}' in {element_element.get('name')}")
    return _eval_number_expression(value_text)


# Fail early when a requested material name is missing from the loaded graph.
def _require_entry(material_name: str, material_library: MaterialLibrary) -> MaterialEntry:
    try:
        return material_library.entries_by_name[material_name]
    except KeyError as error:
        raise KeyError(f"Unknown material '{material_name}'.") from error


def load_material_library(
    elements_xml_path: Path = ELEMENTS_XML_PATH,
    materials_xml_path: Path = MATERIALS_XML_PATH,
) -> MaterialLibrary:
    """Load the primitive and composite material graph from the repo XML definitions."""
    # First collect the primitive element masses and interaction lengths from elements.xml.
    entries_by_name: Dict[str, MaterialEntry] = {}
    atomic_mass_by_symbol: Dict[str, float] = {}

    elements_root = _read_xml_root(elements_xml_path)
    # The <element> entries carry the atomic masses that later turn stoichiometric counts
    # into the mass fractions needed by the interaction-length rule.
    for child_element in elements_root:
        if child_element.tag != "element":
            continue
        element_name = child_element.get("name")
        formula_name = child_element.get("formula")
        if not element_name or not formula_name:
            continue
        atomic_mass = _read_atomic_mass_g_mol(child_element)
        atomic_mass_by_symbol[formula_name] = atomic_mass
        atomic_mass_by_symbol[element_name] = atomic_mass

    # The primitive <material> entries in elements.xml already have tabulated lambda_I values.
    for child_element in elements_root:
        if child_element.tag != "material":
            continue
        primitive_name = child_element.get("name")
        formula_name = child_element.get("formula")
        if not primitive_name:
            continue
        primitive_entry = MaterialEntry(
            name=primitive_name,
            density_g_cm3=_read_density_g_cm3(child_element),
            lambda_I_mm=_read_lambda_I_mm(child_element),
            atomic_mass_g_mol=atomic_mass_by_symbol.get(formula_name or primitive_name),
        )
        entries_by_name[primitive_name] = primitive_entry
        if formula_name:
            entries_by_name[formula_name] = primitive_entry

    # Then load the named mixtures and compounds from materials.xml and attach their pieces.
    materials_root = _read_xml_root(materials_xml_path)
    for material_element in materials_root.findall("material"):
        material_name = material_element.get("name")
        if not material_name:
            continue
        material_entry = MaterialEntry(
            name=material_name,
            density_g_cm3=_read_density_g_cm3(material_element),
        )
        # Keep only the composition-carrying children because constants and comments do not
        # change the material mixture used by the hadronic interaction model.
        for child_element in material_element:
            if child_element.tag not in {"fraction", "composite"}:
                continue
            ref_name = child_element.get("ref")
            amount_text = child_element.get("n")
            if not ref_name or amount_text is None:
                raise ValueError(f"Material '{material_name}' has an incomplete {child_element.tag} entry.")
            material_entry.pieces.append(
                MaterialPiece(
                    name=ref_name,
                    amount=_eval_number_expression(amount_text),
                    amount_kind=child_element.tag,
                )
            )
        entries_by_name[material_name] = material_entry

    return MaterialLibrary(entries_by_name=entries_by_name)


# Stoichiometric materials such as polystyrene are written in integer counts, so resolve
# their effective molar mass before turning them into mass fractions.
def resolve_material_mass_g_mol(
    material_name: str,
    material_library: MaterialLibrary,
) -> float:
    """Resolve the effective molar mass for a stoichiometric material."""
    material_entry = _require_entry(material_name, material_library)
    # Primitive elements already know their mass, so reuse it directly.
    if material_entry.mass_resolved_g_mol is not None:
        return material_entry.mass_resolved_g_mol
    if material_entry.atomic_mass_g_mol is not None:
        material_entry.mass_resolved_g_mol = material_entry.atomic_mass_g_mol
        return material_entry.mass_resolved_g_mol
    if not material_entry.pieces:
        raise ValueError(f"Material '{material_name}' has no stoichiometric mass information.")

    amount_kinds = {piece.amount_kind for piece in material_entry.pieces}
    if amount_kinds != {"composite"}:
        raise ValueError(
            f"Material '{material_name}' is not a pure stoichiometric material and has no molar mass."
        )

    # A stoichiometric material mass is just the sum of each child mass weighted by its count.
    resolved_mass_g_mol = 0.0
    for piece in material_entry.pieces:
        child_mass_g_mol = resolve_material_mass_g_mol(piece.name, material_library)
        resolved_mass_g_mol += piece.amount * child_mass_g_mol
    if resolved_mass_g_mol <= 0.0:
        raise ValueError(f"Material '{material_name}' resolved to a non-positive molar mass.")

    material_entry.mass_resolved_g_mol = resolved_mass_g_mol
    return resolved_mass_g_mol


# Convert one material definition into the mass fractions needed by the hadronic
# interaction-length rule, whether the XML started from fractions or stoichiometric counts.
def _mass_fractions_from_pieces(
    material_name: str,
    material_library: MaterialLibrary,
) -> Dict[str, float]:
    """Convert one material definition into child mass fractions."""
    material_entry = _require_entry(material_name, material_library)
    if not material_entry.pieces:
        raise ValueError(f"Material '{material_name}' has no composition pieces.")

    amount_kinds = {piece.amount_kind for piece in material_entry.pieces}
    if amount_kinds == {"fraction"}:
        # Fraction-defined materials already encode mass weights, so only a normalization is needed.
        fraction_sum = sum(piece.amount for piece in material_entry.pieces)
        if fraction_sum <= 0.0:
            raise ValueError(f"Material '{material_name}' has a non-positive total mass fraction.")
        return {
            piece.name: piece.amount / fraction_sum
            for piece in material_entry.pieces
        }
    if amount_kinds == {"composite"}:
        # Composite-defined materials start from integer counts, so convert them into mass
        # fractions before applying the hadronic mixture rule.
        weighted_masses: Dict[str, float] = {}
        total_mass = 0.0
        for piece in material_entry.pieces:
            child_mass = resolve_material_mass_g_mol(piece.name, material_library)
            child_weight = piece.amount * child_mass
            weighted_masses[piece.name] = child_weight
            total_mass += child_weight
        if total_mass <= 0.0:
            raise ValueError(f"Material '{material_name}' has a non-positive total stoichiometric mass.")
        return {
            child_name: child_weight / total_mass
            for child_name, child_weight in weighted_masses.items()
        }
    raise ValueError(
        f"Material '{material_name}' mixes fraction and composite definitions in one material."
    )


# Resolve one material all the way down to a single nuclear interaction length in mm by
# walking the material graph and combining child materials with the mass-fraction rule.
def resolve_material_lambda_mm(
    material_name: str,
    material_library: MaterialLibrary,
    _active_stack: Set[str] | None = None,
) -> float:
    """Resolve the nuclear interaction length in mm for one material."""
    material_entry = _require_entry(material_name, material_library)
    # Reuse cached results because the same materials appear over and over across layers.
    if material_entry.lambda_I_resolved_mm is not None:
        return material_entry.lambda_I_resolved_mm
    if material_entry.lambda_I_mm is not None:
        material_entry.lambda_I_resolved_mm = material_entry.lambda_I_mm
        return material_entry.lambda_I_resolved_mm

    # Track the active recursion chain so a malformed XML loop fails clearly.
    if _active_stack is None:
        _active_stack = set()
    if material_name in _active_stack:
        cycle_text = " -> ".join([*_active_stack, material_name])
        raise ValueError(f"Material recursion cycle detected: {cycle_text}")
    _active_stack.add(material_name)

    try:
        # Once the child mass fractions are known, the material lambda_I follows from
        # 1 / lambda_I_eff = sum(w_i / lambda_I_i).
        mass_fractions = _mass_fractions_from_pieces(material_name, material_library)
        inverse_lambda_mm = 0.0
        # Resolve each child first, then accumulate the inverse interaction length for the mixture.
        for child_name, mass_fraction in mass_fractions.items():
            child_lambda_mm = resolve_material_lambda_mm(child_name, material_library, _active_stack)
            if child_lambda_mm <= 0.0:
                raise ValueError(f"Material '{child_name}' resolved to a non-positive lambda_I.")
            inverse_lambda_mm += mass_fraction / child_lambda_mm
    finally:
        _active_stack.remove(material_name)

    if inverse_lambda_mm <= 0.0:
        raise ValueError(f"Material '{material_name}' resolved to a non-positive inverse lambda_I.")
    material_entry.lambda_I_resolved_mm = 1.0 / inverse_lambda_mm
    return material_entry.lambda_I_resolved_mm


# The default CLI resolves the current HCAL materials so the lookup can be checked quickly
# before the full geometry analysis uses it.
def main() -> int:
    arguments = parse_arguments()
    material_library = load_material_library()

    # With no explicit names, print the materials that the current generated HCAL geometries use.
    material_names = arguments.material if arguments.material else [
        "Air",
        "Polystyrene",
        "StainlessSteelSAE304",
    ]
    for material_name in material_names:
        lambda_value_mm = resolve_material_lambda_mm(material_name, material_library)
        print(f"{material_name}: {lambda_value_mm:.6f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
