#!/usr/bin/env python3
"""Resolve nuclear interaction lengths for materials used by the HCAL study."""

from __future__ import annotations

import argparse
import ast
import math
from dataclasses import dataclass, field
from pathlib import Path
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

# One material component from the XML graph.
@dataclass
class MaterialPiece:
    name: str
    amount: float
    amount_kind: str


# One loaded material entry with optional composition pieces.
@dataclass
class MaterialEntry:
    name: str
    density_g_cm3: float | None = None
    lambda_I_mm: float | None = None
    atomic_mass_g_mol: float | None = None
    pieces: list[MaterialPiece] = field(default_factory=list)
    lambda_I_resolved_mm: float | None = None
    mass_resolved_g_mol: float | None = None


# One container for the loaded material graph.
@dataclass
class MaterialLibrary:
    entries_by_name: dict[str, MaterialEntry]


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


def _read_xml_root(xml_path: Path) -> xml_tree.Element:
    """Read one XML material definition file."""
    if not xml_path.exists():
        raise FileNotFoundError(f"Material definition file not found: {xml_path}")
    return xml_tree.parse(xml_path).getroot()


def _eval_expression_node(node: ast.AST) -> float:
    """Evaluate one restricted numeric AST node."""
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


def _require_entry(material_name: str, material_library: MaterialLibrary) -> MaterialEntry:
    """Look up one material entry by name."""
    try:
        return material_library.entries_by_name[material_name]
    except KeyError as error:
        raise KeyError(f"Unknown material '{material_name}'.") from error


def load_material_library(
    elements_xml_path: Path = ELEMENTS_XML_PATH,
    materials_xml_path: Path = MATERIALS_XML_PATH,
) -> MaterialLibrary:
    """Load the primitive and composite material graph from the repo XML definitions."""
    # Collect the primitive element masses from elements.xml first.
    entries_by_name: dict[str, MaterialEntry] = {}
    atomic_mass_by_symbol: dict[str, float] = {}

    elements_root = _read_xml_root(elements_xml_path)
    # Read atomic masses from the <element> entries.
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

    # Read primitive material entries that already carry tabulated lambda_I values.
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

    # Load named mixtures and compounds from materials.xml.
    materials_root = _read_xml_root(materials_xml_path)
    for material_element in materials_root.findall("material"):
        material_name = material_element.get("name")
        if not material_name:
            continue
        material_entry = MaterialEntry(
            name=material_name,
            density_g_cm3=_read_density_g_cm3(material_element),
        )
        # Keep only the child entries that define the material composition.
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


def resolve_material_mass_g_mol(
    material_name: str,
    material_library: MaterialLibrary,
) -> float:
    """Resolve the effective molar mass for a stoichiometric material."""
    material_entry = _require_entry(material_name, material_library)
    # Reuse a cached or direct atomic mass when it is already available.
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

    # Sum each child molar mass weighted by its stoichiometric count.
    resolved_mass_g_mol = 0.0
    for piece in material_entry.pieces:
        child_mass_g_mol = resolve_material_mass_g_mol(piece.name, material_library)
        resolved_mass_g_mol += piece.amount * child_mass_g_mol
    if resolved_mass_g_mol <= 0.0:
        raise ValueError(f"Material '{material_name}' resolved to a non-positive molar mass.")

    material_entry.mass_resolved_g_mol = resolved_mass_g_mol
    return resolved_mass_g_mol


def _mass_fractions_from_pieces(
    material_name: str,
    material_library: MaterialLibrary,
) -> dict[str, float]:
    """Convert one material definition into child mass fractions."""
    material_entry = _require_entry(material_name, material_library)
    if not material_entry.pieces:
        raise ValueError(f"Material '{material_name}' has no composition pieces.")

    amount_kinds = {piece.amount_kind for piece in material_entry.pieces}
    if amount_kinds == {"fraction"}:
        # Normalize fraction-defined materials directly.
        fraction_sum = sum(piece.amount for piece in material_entry.pieces)
        if fraction_sum <= 0.0:
            raise ValueError(f"Material '{material_name}' has a non-positive total mass fraction.")
        return {
            piece.name: piece.amount / fraction_sum
            for piece in material_entry.pieces
        }
    if amount_kinds == {"composite"}:
        # Convert stoichiometric counts into child mass fractions.
        weighted_masses: dict[str, float] = {}
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


def resolve_material_lambda_mm(
    material_name: str,
    material_library: MaterialLibrary,
    active_material_names: set[str] | None = None,
) -> float:
    """Resolve the nuclear interaction length in mm for one material."""
    material_entry = _require_entry(material_name, material_library)
    # Reuse cached values whenever the material was already resolved.
    if material_entry.lambda_I_resolved_mm is not None:
        return material_entry.lambda_I_resolved_mm
    if material_entry.lambda_I_mm is not None:
        material_entry.lambda_I_resolved_mm = material_entry.lambda_I_mm
        return material_entry.lambda_I_resolved_mm

    # Track the active recursion path so XML cycles fail clearly.
    if active_material_names is None:
        active_material_names = set()
    if material_name in active_material_names:
        cycle_text = " -> ".join([*active_material_names, material_name])
        raise ValueError(f"Material recursion cycle detected: {cycle_text}")
    active_material_names.add(material_name)

    try:
        # Apply the mass-fraction mixture rule to the child materials.
        mass_fractions = _mass_fractions_from_pieces(material_name, material_library)
        inverse_lambda_mm = 0.0
        # Resolve each child material before accumulating the inverse lambda_I.
        for child_name, mass_fraction in mass_fractions.items():
            child_lambda_mm = resolve_material_lambda_mm(
                child_name,
                material_library,
                active_material_names,
            )
            if child_lambda_mm <= 0.0:
                raise ValueError(f"Material '{child_name}' resolved to a non-positive lambda_I.")
            inverse_lambda_mm += mass_fraction / child_lambda_mm
    finally:
        active_material_names.remove(material_name)

    if inverse_lambda_mm <= 0.0:
        raise ValueError(f"Material '{material_name}' resolved to a non-positive inverse lambda_I.")
    material_entry.lambda_I_resolved_mm = 1.0 / inverse_lambda_mm
    return material_entry.lambda_I_resolved_mm


def main() -> int:
    """Resolve requested materials and print their lambda_I values."""
    arguments = parse_arguments()
    material_library = load_material_library()

    # Fall back to the main HCAL study materials when no names are provided.
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
