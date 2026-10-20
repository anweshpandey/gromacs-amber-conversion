#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from collections import defaultdict


# ============================================================================
# HARD-CODED PATHS
# ============================================================================

GROMACS_RTP = Path(
    "/home/apandey/softwares/gromacs-5.1.1/gromacs/"
    "gromacs-5.1.1/share/top/amber99sb.ff/aminoacids.rtp"
)

AMBER_LIBRARY = Path(
    "/home/apandey/softwares/TITAN-master/titan/amber_library.json"
)

MAPPING_JSON = Path(
    "/home/apandey/softwares/my_scripts/gromacs_to_amber_atom_mapping.json"
)



# ============================================================================
# DEFAULTS
# ============================================================================

DEFAULT_OUTPUT_PDB = "min_amber.pdb"


# ============================================================================
# PDB HELPERS
# ============================================================================

def is_atom_line(line):
    return line.startswith(("ATOM  ", "HETATM"))


def atom_name(line):
    return line[12:16].strip()


def residue_name(line):
    return line[17:20].strip()


def residue_number(line):
    return line[22:26].strip()


def insertion_code(line):
    return line[26:27].strip()


def chain_id(line):
    return line[21:22].strip() if len(line) >= 22 else ""


def serial_number(line):
    try:
        return int(line[6:11].strip())
    except ValueError:
        return line[6:11].strip()


def coordinates(line):
    try:
        return (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
    except (ValueError, IndexError):
        return None


def element(line):
    return line[76:78].strip() if len(line) >= 78 else ""


def replace_atom_name(line, new_name):
    if len(new_name) > 4:
        raise ValueError(
            f"Amber atom name '{new_name}' is longer than 4 characters."
        )

    return line[:12] + f"{new_name:>4}" + line[16:]


def replace_residue_name(line, new_resname):
    if len(new_resname) > 3:
        raise ValueError(
            f"Amber residue name '{new_resname}' is longer than "
            "3 characters."
        )

    return line[:17] + f"{new_resname:<3}" + line[20:]


def atom_to_hetatm(line):
    if line.startswith("ATOM  "):
        return "HETATM" + line[6:]
    return line


# ============================================================================
# JSON LOADERS
# ============================================================================

def load_json(path, description):
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"{description} must contain a JSON dictionary:\n{path}"
        )

    return data


# ============================================================================
# AMBER LIBRARY
# ============================================================================

def amber_atoms_by_residue(amber_library):
    """
    Convert:
        {"NMET HB2": charge, "NMET HB3": charge, ...}

    into:
        {"NMET": {"HB2", "HB3", ...}}
    """
    result = defaultdict(set)

    for key in amber_library:
        parts = key.split(maxsplit=1)

        if len(parts) != 2:
            continue

        resname, aname = parts
        result[resname].add(aname)

    return result


# ============================================================================
# MAPPING
# ============================================================================

def get_residue_mapping(mapping, resname, is_nterm=False, is_cterm=False):
    """
    Mapping is expected to be:

        {
            "NMET": {
                "HB1": "HB2",
                "HB2": "HB3",
                ...
            }
        }

    AMBER uses distinct residue templates for the N-terminal and
    C-terminal forms of an amino acid (e.g. "NMET", "CMET") because
    they carry extra/missing atoms compared to the mid-chain form
    ("MET"). The PDB residue name itself always stays "MET" - only the
    mapping/library lookup key changes.

    Returns a (residue_mapping, resolved_resname) tuple. resolved_resname
    is the key that was actually used ("NMET"/"CMET"/resname), which the
    caller needs to build the correct Amber-library key.
    """
    if is_nterm and f"N{resname}" in mapping:
        nkey = f"N{resname}"
        return mapping[nkey], nkey

    if is_cterm and f"C{resname}" in mapping:
        ckey = f"C{resname}"
        return mapping[ckey], ckey

    return mapping.get(resname), resname


def normalize_legacy_hydrogen_name(atom_name):
    """
    Some GROMACS PDB output uses the legacy PDBv2 convention of writing
    the branch digit BEFORE the element/atom name (e.g. "1HH1", "2HH1")
    instead of the PDBv3/Amber convention of writing it AFTER
    ("HH11", "HH12").

        "1HH1" -> "HH11"
        "2HH1" -> "HH12"
        "1HH2" -> "HH21"
        "2HH2" -> "HH22"

    Returns None if the name doesn't start with a digit (i.e. there is
    nothing to normalize).
    """
    if atom_name and atom_name[0].isdigit():
        return atom_name[1:] + atom_name[0]

    return None


def compute_terminal_residues(lines, ligand_resname, ligand_resid):
    """
    Identify the first and last residue of each contiguous protein chain
    (chains are split on TER records and on chain-ID changes) and flag
    them as N-terminal / C-terminal respectively. Ligand atoms are
    ignored - they neither belong to nor break a protein chain segment.

    Returns:
        {(resname, resid, icode): {"nterm": bool, "cterm": bool}, ...}
    """

    def is_ligand(resname, resid):
        return (
            resname == ligand_resname
            and (ligand_resid is None or resid == ligand_resid)
        )

    segments = []
    current = []
    prev_chain = None

    for line in lines:

        if line.startswith("TER"):
            if current:
                segments.append(current)
                current = []
            prev_chain = None
            continue

        if not is_atom_line(line):
            continue

        resname = residue_name(line)
        resid = residue_number(line)
        icode = insertion_code(line)
        cid = chain_id(line)

        if is_ligand(resname, resid):
            continue

        if prev_chain is not None and cid != prev_chain and current:
            segments.append(current)
            current = []

        prev_chain = cid

        key = (resname, resid, icode)

        if not current or current[-1] != key:
            current.append(key)

    if current:
        segments.append(current)

    flags = {}

    for segment in segments:
        if not segment:
            continue
        flags.setdefault(segment[0], {})["nterm"] = True
        flags.setdefault(segment[-1], {})["cterm"] = True

    return flags


def collect_residue_atoms(lines, ligand_resname, ligand_resid):
    """
    Build {(resname, resid, icode): {atom names present}} for every
    non-ligand residue instance in the PDB. Used to resolve residues
    whose Amber identity depends on which atoms are actually present
    (e.g. histidine protonation state), not just on the PDB resname.
    """

    def is_ligand(resname, resid):
        return (
            resname == ligand_resname
            and (ligand_resid is None or resid == ligand_resid)
        )

    atoms = defaultdict(set)

    for line in lines:
        if not is_atom_line(line):
            continue

        resname = residue_name(line)
        resid = residue_number(line)

        if is_ligand(resname, resid):
            continue

        icode = insertion_code(line)
        atoms[(resname, resid, icode)].add(atom_name(line))

    return atoms


def resolve_histidine_tautomer(atoms_present):
    """
    GROMACS keeps the PDB residue name "HIS" for all histidines
    regardless of protonation state - the tautomer choice only lives
    in the topology, not the coordinates. Amber instead needs the
    explicit residue name (HID/HIE/HIP), so we infer it from which
    delta/epsilon hydrogen is actually present:

        HD1 present only -> HID (protonated on ND1)
        HE2 present only -> HIE (protonated on NE2)
        both present      -> HIP (doubly protonated / positively charged)

    Returns None if neither hydrogen is present (cannot be resolved).
    """
    has_hd1 = "HD1" in atoms_present
    has_he2 = "HE2" in atoms_present

    if has_hd1 and has_he2:
        return "HIP"
    if has_hd1:
        return "HID"
    if has_he2:
        return "HIE"

    return None


def resolve_ambiguous_residue(resname, atoms_present):
    """
    Some PDB residue names don't map 1:1 onto an Amber residue template;
    the correct template depends on which atoms are present. Returns
    (effective_resname, new_pdb_resname) where:

        effective_resname - what to use for mapping/library lookups
        new_pdb_resname   - if not None, the resname to write into the
                             output PDB (3 chars, no N/C prefix)

    For residues with no ambiguity, returns (resname, None) unchanged.
    """
    if resname == "HIS":
        tautomer = resolve_histidine_tautomer(atoms_present)
        if tautomer is not None:
            return tautomer, tautomer

    return resname, None


# ============================================================================
# PDB CONVERSION
# ============================================================================

def convert_pdb(
    input_pdb,
    output_pdb,
    removed_file,
    comparison_file,
    mapping_report_file,
    mapping,
    amber_library,
    ligand_resname="LIG",
    ligand_resid=None,
):

    with open(input_pdb, "r") as f:
        lines = f.readlines()

    amber_atoms = amber_atoms_by_residue(amber_library)

    terminal_flags = compute_terminal_residues(
        lines, ligand_resname, ligand_resid
    )

    residue_atoms = collect_residue_atoms(
        lines, ligand_resname, ligand_resid
    )

    input_records = []
    output_records = []
    removed_records = []
    mapping_records = []

    output_lines = []

    renamed = 0
    unchanged = 0
    ligand_count = 0

    # ------------------------------------------------------------------------
    # FIRST PASS:
    # map every protein atom and reject anything that cannot produce a
    # valid Amber-library atom.
    # ------------------------------------------------------------------------

    for line_no, line in enumerate(lines, start=1):

        if not is_atom_line(line):
            output_lines.append(line)
            continue

        resname = residue_name(line)
        resid = residue_number(line)
        icode = insertion_code(line)
        old_atom = atom_name(line)

        record = {
            "line": line_no,
            "serial": serial_number(line),
            "resname": resname,
            "resid": resid,
            "icode": icode,
            "atom": old_atom,
            "xyz": coordinates(line),
            "element": element(line),
        }

        input_records.append(record)

        # --------------------------------------------------------------------
        # LIGAND: preserve it.
        # --------------------------------------------------------------------

        if (
            resname == ligand_resname
            and (ligand_resid is None or resid == ligand_resid)
        ):
            new_line = atom_to_hetatm(line)
            output_lines.append(new_line)

            output_records.append({
                **record,
                "new_atom": old_atom,
                "action": "LIGAND_PRESERVED",
            })

            ligand_count += 1
            continue

        # --------------------------------------------------------------------
        # RESIDUE MUST EXIST IN MAPPING
        #
        # Some PDB residue names (e.g. "HIS") don't correspond 1:1 to an
        # Amber residue template - the correct template depends on which
        # atoms are actually present. Resolve that first.
        # --------------------------------------------------------------------

        effective_resname, new_pdb_resname = resolve_ambiguous_residue(
            resname, residue_atoms.get((resname, resid, icode), set())
        )

        flags = terminal_flags.get((resname, resid, icode), {})

        residue_mapping, resolved_resname = get_residue_mapping(
            mapping,
            effective_resname,
            is_nterm=flags.get("nterm", False),
            is_cterm=flags.get("cterm", False),
        )

        if residue_mapping is None:
            reason = "Residue absent from mapping JSON"

            if resname == "HIS" and new_pdb_resname is None:
                reason = (
                    "Histidine protonation state (HID/HIE/HIP) could "
                    "not be determined: neither HD1 nor HE2 is present "
                    "on this residue"
                )

            removed_records.append({
                **record,
                "mapped_atom": "",
                "reason": reason,
            })
            continue

        # --------------------------------------------------------------------
        # INPUT GROMACS ATOM MUST EXIST IN MAPPING
        #
        # Try the atom name as-is first. If that fails, retry using the
        # PDBv3/Amber form of legacy leading-digit hydrogen names (e.g.
        # "1HH1" -> "HH11"), since GROMACS sometimes emits the older
        # PDBv2-style naming that the mapping JSON doesn't use directly.
        # --------------------------------------------------------------------

        lookup_atom = old_atom

        if lookup_atom not in residue_mapping:
            normalized = normalize_legacy_hydrogen_name(old_atom)

            if normalized is not None and normalized in residue_mapping:
                lookup_atom = normalized
            else:
                removed_records.append({
                    **record,
                    "mapped_atom": "",
                    "reason": (
                        "Input GROMACS atom absent from mapping "
                        "for this residue"
                    ),
                })
                continue

        new_atom = str(residue_mapping[lookup_atom])
        amber_key = f"{resolved_resname} {new_atom}"

        # --------------------------------------------------------------------
        # AMBER LIBRARY IS THE FINAL AUTHORITY
        # --------------------------------------------------------------------

        if amber_key not in amber_library:
            removed_records.append({
                **record,
                "mapped_atom": new_atom,
                "reason": (
                    "Mapped atom is absent from Amber library"
                ),
            })
            continue

        # Valid atom
        new_line = line

        if new_pdb_resname is not None:
            new_line = replace_residue_name(new_line, new_pdb_resname)

        new_line = replace_atom_name(new_line, new_atom)
        output_lines.append(new_line)

        action = "RENAMED" if old_atom != new_atom else "UNCHANGED"

        if action == "RENAMED":
            renamed += 1
        else:
            unchanged += 1

        output_records.append({
            **record,
            "new_atom": new_atom,
            "action": action,
        })

        mapping_records.append({
            **record,
            "new_atom": new_atom,
            "action": action,
        })

    # ------------------------------------------------------------------------
    # SECOND PASS:
    # check that two input atoms have not mapped onto the same Amber atom
    # within the same residue.
    #
    # If they have, KEEP THE FIRST occurrence and remove subsequent ones.
    # ------------------------------------------------------------------------

    seen = set()
    final_lines = []
    final_output_records = []

    # output_records is used to identify which atom lines survive.
    output_record_iter = iter(output_records)

    for line in output_lines:

        if not is_atom_line(line):
            final_lines.append(line)
            continue

        resname = residue_name(line)
        resid = residue_number(line)
        icode = insertion_code(line)
        aname = atom_name(line)

        # Ligands are not checked against Amber residue atom uniqueness.
        if (
            resname == ligand_resname
            and (ligand_resid is None or resid == ligand_resid)
        ):
            final_lines.append(line)
            continue

        key = (resname, resid, icode, aname)

        if key in seen:

            rec = {
                "line": None,
                "serial": serial_number(line),
                "resname": resname,
                "resid": resid,
                "icode": icode,
                "atom": aname,
                "mapped_atom": aname,
                "xyz": coordinates(line),
                "element": element(line),
                "reason": (
                    "Duplicate Amber atom name within residue "
                    "after GROMACS -> Amber mapping"
                ),
            }

            removed_records.append(rec)
            continue

        seen.add(key)
        final_lines.append(line)

    output_lines = final_lines

    # ------------------------------------------------------------------------
    # WRITE OUTPUT PDB
    # ------------------------------------------------------------------------

    with open(output_pdb, "w") as f:
        f.writelines(output_lines)

    # ------------------------------------------------------------------------
    # READ OUTPUT AGAIN
    # ------------------------------------------------------------------------

    with open(output_pdb, "r") as f:
        final_pdb_lines = f.readlines()

    final_records = []

    for line_no, line in enumerate(final_pdb_lines, start=1):

        if not is_atom_line(line):
            continue

        final_records.append({
            "line": line_no,
            "serial": serial_number(line),
            "resname": residue_name(line),
            "resid": residue_number(line),
            "icode": insertion_code(line),
            "atom": atom_name(line),
            "xyz": coordinates(line),
            "element": element(line),
        })

    # ------------------------------------------------------------------------
    # FINAL AMBER VALIDATION
    # ------------------------------------------------------------------------

    validation_failures = []

    for rec in final_records:

        resname = rec["resname"]
        resid = rec["resid"]

        if (
            resname == ligand_resname
            and (ligand_resid is None or resid == ligand_resid)
        ):
            continue

        key = f"{resname} {rec['atom']}"

        if key not in amber_library:
            validation_failures.append(rec)

    # ------------------------------------------------------------------------
    # DETERMINE EXPECTED AMBER ATOMS FOR EACH RESIDUE PRESENT IN INPUT
    #
    # We use the mapping and Amber library to determine the valid reference
    # atom set. We do NOT invent atoms in the PDB.
    # ------------------------------------------------------------------------

    expected = defaultdict(set)
    actual = defaultdict(set)

    for rec in input_records:

        resname = rec["resname"]
        resid = rec["resid"]
        icode = rec["icode"]

        if (
            resname == ligand_resname
            and (ligand_resid is None or resid == ligand_resid)
        ):
            continue

        effective_resname, new_pdb_resname = resolve_ambiguous_residue(
            resname, residue_atoms.get((resname, resid, icode), set())
        )

        output_resname = (
            new_pdb_resname if new_pdb_resname is not None else resname
        )

        flags = terminal_flags.get((resname, resid, icode), {})

        residue_mapping, resolved_resname = get_residue_mapping(
            mapping,
            effective_resname,
            is_nterm=flags.get("nterm", False),
            is_cterm=flags.get("cterm", False),
        )

        if residue_mapping is None:
            continue

        for mapped_atom in residue_mapping.values():

            if f"{resolved_resname} {mapped_atom}" in amber_library:
                expected[(output_resname, resid, icode)].add(mapped_atom)

    for rec in final_records:

        resname = rec["resname"]
        resid = rec["resid"]
        icode = rec["icode"]

        if (
            resname == ligand_resname
            and (ligand_resid is None or resid == ligand_resid)
        ):
            continue

        actual[(resname, resid, icode)].add(rec["atom"])

    missing = []

    for residue_key, expected_atoms in expected.items():

        for missing_atom in sorted(
            expected_atoms - actual.get(residue_key, set())
        ):
            missing.append({
                "resname": residue_key[0],
                "resid": residue_key[1],
                "icode": residue_key[2],
                "atom": missing_atom,
            })

    # ------------------------------------------------------------------------
    # REMOVED ATOMS TXT
    # ------------------------------------------------------------------------

    with open(removed_file, "w") as f:

        f.write("=" * 100 + "\n")
        f.write("REMOVED / REJECTED ATOMS\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Input : {input_pdb}\n")
        f.write(f"Output: {output_pdb}\n")
        f.write(f"Total : {len(removed_records)}\n\n")

        if not removed_records:
            f.write("NONE\n")

        for i, rec in enumerate(removed_records, 1):

            f.write(
                f"[{i}]\n"
                f"  Residue      : {rec['resname']} "
                f"{rec['resid']}{rec['icode']}\n"
                f"  Serial       : {rec['serial']}\n"
                f"  Input atom   : {rec['atom']}\n"
                f"  Mapped atom  : {rec.get('mapped_atom', '')}\n"
                f"  Coordinates  : {rec['xyz']}\n"
                f"  Element      : {rec['element']}\n"
                f"  Reason       : {rec['reason']}\n\n"
            )

    # ------------------------------------------------------------------------
    # COMPARISON REPORT
    # ------------------------------------------------------------------------

    with open(comparison_file, "w") as f:

        f.write("=" * 100 + "\n")
        f.write("INPUT PDB vs OUTPUT PDB COMPARISON\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Input PDB       : {input_pdb}\n")
        f.write(f"Output PDB      : {output_pdb}\n")
        f.write(f"Mapping JSON    : {MAPPING_JSON}\n")
        f.write(f"Amber library   : {AMBER_LIBRARY}\n\n")

        f.write("-" * 100 + "\n")
        f.write("SUMMARY\n")
        f.write("-" * 100 + "\n")

        f.write(f"Input atom count          : {len(input_records)}\n")
        f.write(f"Output atom count         : {len(final_records)}\n")
        f.write(f"Removed atom count        : {len(removed_records)}\n")
        f.write(f"Renamed atom count        : {renamed}\n")
        f.write(f"Unchanged atom count      : {unchanged}\n")
        f.write(f"Ligand atom count         : {ligand_count}\n")
        f.write(f"Missing expected atoms    : {len(missing)}\n")
        f.write(f"Final library failures    : {len(validation_failures)}\n")

        f.write("\n")
        f.write("-" * 100 + "\n")
        f.write("MAPPING / INPUT -> OUTPUT\n")
        f.write("-" * 100 + "\n\n")

        for rec in mapping_records:
            f.write(
                f"{rec['resname']:<5} "
                f"{rec['resid']:<5} "
                f"{rec['atom']:<5} -> "
                f"{rec['new_atom']:<5} "
                f"{rec['action']}\n"
            )

        f.write("\n")
        f.write("-" * 100 + "\n")
        f.write("REMOVED ATOMS\n")
        f.write("-" * 100 + "\n\n")

        if removed_records:
            for rec in removed_records:
                f.write(
                    f"{rec['resname']:<5} "
                    f"{rec['resid']:<5} "
                    f"{rec['atom']:<5} -> "
                    f"{rec.get('mapped_atom', ''):<5} | "
                    f"{rec['reason']}\n"
                )
        else:
            f.write("NONE\n")

        f.write("\n")
        f.write("-" * 100 + "\n")
        f.write("MISSING EXPECTED AMBER ATOMS\n")
        f.write("-" * 100 + "\n\n")

        if missing:
            for rec in missing:
                f.write(
                    f"{rec['resname']:<5} "
                    f"{rec['resid']:<5} "
                    f"{rec['atom']:<5}\n"
                )
        else:
            f.write("NONE\n")

        f.write("\n")
        f.write("-" * 100 + "\n")
        f.write("FINAL AMBER LIBRARY VALIDATION\n")
        f.write("-" * 100 + "\n\n")

        if validation_failures:
            f.write(
                f"FAILED: {len(validation_failures)} invalid atoms remain.\n"
            )
            for rec in validation_failures:
                f.write(
                    f"{rec['resname']} {rec['resid']} "
                    f"{rec['atom']}\n"
                )
        else:
            f.write(
                "PASSED: every non-ligand atom in the output PDB "
                "exists in amber_library.json.\n"
            )

    # ------------------------------------------------------------------------
    # MAPPING REPORT
    # ------------------------------------------------------------------------

    with open(mapping_report_file, "w") as f:

        f.write("=" * 100 + "\n")
        f.write("GROMACS -> AMBER MAPPING VALIDATION\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"GROMACS RTP    : {GROMACS_RTP}\n")
        f.write(f"Amber library  : {AMBER_LIBRARY}\n")
        f.write(f"Mapping JSON   : {MAPPING_JSON}\n\n")

        for resname in sorted(mapping):

            f.write(f"[{resname}]\n")

            for old_atom, new_atom in mapping[resname].items():

                key = f"{resname} {new_atom}"

                status = (
                    "VALID"
                    if key in amber_library
                    else "NOT IN AMBER LIBRARY"
                )

                f.write(
                    f"{old_atom:<6} -> "
                    f"{str(new_atom):<6} "
                    f"[{status}]\n"
                )

            f.write("\n")

    return {
        "input_count": len(input_records),
        "output_count": len(final_records),
        "removed": len(removed_records),
        "renamed": renamed,
        "unchanged": unchanged,
        "ligand": ligand_count,
        "missing": len(missing),
        "validation_failures": len(validation_failures),
        "removed_file": removed_file,
        "comparison_file": comparison_file,
        "mapping_report_file": mapping_report_file,
    }




# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("GROMACS -> AMBER PDB CONVERSION PIPELINE")
    print("=" * 80)

    print("\nHard-coded reference files:")
    print(f"  GROMACS RTP   : {GROMACS_RTP}")
    print(f"  Amber library : {AMBER_LIBRARY}")
    print(f"  Mapping JSON  : {MAPPING_JSON}")

    # ------------------------------------------------------------------------
    # Validate reference files
    # ------------------------------------------------------------------------

    for path, description in [
        (GROMACS_RTP, "GROMACS aminoacids.rtp"),
        (AMBER_LIBRARY, "Amber library"),
        (MAPPING_JSON, "GROMACS -> Amber mapping"),
    ]:
        if not path.exists():
            print(f"\nERROR: {description} not found:")
            print(path)
            sys.exit(1)

    # ------------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------------

    input_value = input("\nInput PDB file: ").strip()

    if not input_value:
        print("ERROR: No input PDB supplied.")
        sys.exit(1)

    input_pdb = Path(input_value).expanduser()

    if not input_pdb.is_file():
        print(f"ERROR: Input PDB not found:\n{input_pdb}")
        sys.exit(1)

    output_value = input(
        f"Output PDB [{DEFAULT_OUTPUT_PDB}]: "
    ).strip()

    output_pdb = Path(
        output_value or DEFAULT_OUTPUT_PDB
    ).expanduser()

    ligand_resname = input(
        "Ligand residue name [MOL]: "
    ).strip() or "MOL"

    ligand_resid_input = input(
        "Ligand residue number [ALL]: "
    ).strip()

    ligand_resid = (
        ligand_resid_input
        if ligand_resid_input
        else None
    )

    # ------------------------------------------------------------------------
    # Load reference data
    # ------------------------------------------------------------------------

    mapping = load_json(
        MAPPING_JSON,
        "GROMACS -> Amber mapping",
    )

    amber_library = load_json(
        AMBER_LIBRARY,
        "Amber library",
    )

    # ------------------------------------------------------------------------
    # Generate report filenames
    # ------------------------------------------------------------------------

    removed_file = output_pdb.with_name(
        output_pdb.stem + "_removed_atoms.txt"
    )

    comparison_file = output_pdb.with_name(
        output_pdb.stem + "_comparison.txt"
    )

    mapping_report_file = output_pdb.with_name(
        output_pdb.stem + "_mapping_report.txt"
    )

    # ------------------------------------------------------------------------
    # Convert
    # ------------------------------------------------------------------------

    summary = convert_pdb(
        input_pdb=input_pdb,
        output_pdb=output_pdb,
        removed_file=removed_file,
        comparison_file=comparison_file,
        mapping_report_file=mapping_report_file,
        mapping=mapping,
        amber_library=amber_library,
        ligand_resname=ligand_resname,
        ligand_resid=ligand_resid,
    )

    print("\n" + "=" * 60)
    print("PDB CONVERSION COMPLETE")
    print("=" * 60)

    print(f"Input atoms             : {summary['input_count']}")
    print(f"Output atoms            : {summary['output_count']}")
    print(f"Renamed                 : {summary['renamed']}")
    print(f"Unchanged               : {summary['unchanged']}")
    print(f"Removed                 : {summary['removed']}")
    print(f"Ligand atoms            : {summary['ligand']}")
    print(f"Missing expected atoms : {summary['missing']}")
    print(
        f"Final library failures : "
        f"{summary['validation_failures']}"
    )

    print("\nGenerated:")
    print(f"  {output_pdb}")
    print(f"  {removed_file}")
    print(f"  {comparison_file}")
    print(f"  {mapping_report_file}")

    if summary["validation_failures"] != 0:
        print()
        print("ERROR: Output PDB still contains atoms not recognized by Amber library.")
        print("Please inspect the comparison and removed-atom reports.")
        sys.exit(2)

    print("\n" + "=" * 80)
    print("PDB CONVERSION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Output PDB: {output_pdb}")
    print(f"Removed atoms report: {removed_file}")
    print(f"Comparison report: {comparison_file}")
    print(f"Mapping report: {mapping_report_file}")


if __name__ == "__main__":
    main()
