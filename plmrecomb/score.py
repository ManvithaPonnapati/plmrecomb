from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from Bio import pairwise2
from Bio.Align import substitution_matrices
from Bio.PDB import PDBParser


THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

PARENT_ALIASES: Dict[str, List[str]] = {
    "C1C2":       ["C1C2", "c1c2"],
    "CsChrimson": ["CsChrimson", "CsChR-Chrimson", "CsChrim", "CsChrimson1"],
    "CheRiff":    ["CheRiff"],
}


def clean_seq(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return "".join(s.strip().split()).replace("-", "").upper()


def seq_key(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()[:16]


@dataclass
class ESMClient:
    base_url: str
    timeout: int = 600

    def _post(self, endpoint: str, sequence: str) -> Dict:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        r = requests.post(url, json={"sequence": sequence}, timeout=self.timeout)
        try:
            return r.json()
        except Exception:
            return {"success": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}

    def predict(self, sequence: str) -> Dict:
        return self._post("predict", sequence)

    def embed(self, sequence: str) -> Dict:
        return self._post("embed", sequence)


@dataclass
class MPNNClient:
    base_url: str
    timeout: int = 600

    def sample(
        self,
        pdb_string: str,
        chains: str = "A",
        temp: float = 0.1,
        batch: int = 8,
        fix_pos: str = "",
        inverse: bool = False,
    ) -> List[Dict]:
        url = f"{self.base_url.rstrip('/')}/sample"
        payload = {
            "pdb_string": pdb_string,
            "params": {
                "chains": chains,
                "temp": temp,
                "batch": batch,
                "fix_pos": fix_pos,
                "inverse": inverse,
            },
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        return r.json()


def get_plddt(
    client: ESMClient, sequence: str, cache_dir: Path
) -> Tuple[Optional[float], Optional[str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = seq_key(sequence)
    j = cache_dir / f"{key}.json"
    pdb = cache_dir / f"{key}.pdb"
    if j.exists():
        import json
        rec = json.loads(j.read_text())
        pdb_str = pdb.read_text() if pdb.exists() else None
        return rec["mean_plddt"], pdb_str
    res = client.predict(sequence)
    if not res.get("success"):
        return None, None
    import json
    j.write_text(json.dumps({"mean_plddt": res["mean_plddt"], "length": res.get("length")}))
    if "pdb_str" in res:
        pdb.write_text(res["pdb_str"])
    return res["mean_plddt"], res.get("pdb_str")


def get_embedding(client: ESMClient, sequence: str, cache_dir: Path) -> Optional[np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = seq_key(sequence)
    f = cache_dir / f"{key}.npy"
    if f.exists():
        return np.load(f)
    res = client.embed(sequence)
    if not res.get("success"):
        return None
    emb = np.array(res["embedding"], dtype=np.float32)
    np.save(f, emb)
    return emb


PEAK_THRESHOLD = 0.1


def load_dataset1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    photo_col = next(c for c in df.columns if c.lower().startswith("photocurrent"))
    df = df.rename(columns={photo_col: "photocurrent"})
    df["sequence"] = df["Amino_acid_sequence"].map(clean_seq)
    df["functional"] = (df["photocurrent"] > PEAK_THRESHOLD).astype(int)
    df["source"] = "dataset1"
    return df[["ChR_name", "sequence", "functional", "photocurrent", "source"]] \
        .dropna(subset=["sequence"]).query("sequence != ''")


def load_dataset2(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    peak_cols = [c for c in df.columns if "peak" in c.lower() and "ss" not in c.lower()]
    df["max_peak_any"] = df[peak_cols].max(axis=1, skipna=True)
    df["sequence"] = df["Amino_acid_sequence"].map(clean_seq)
    df["functional"] = (df["max_peak_any"] > PEAK_THRESHOLD).astype(int)
    df = df.rename(columns={"max_peak_any": "photocurrent"})
    df["source"] = "dataset2"
    return df[["ChR_name", "sequence", "functional", "photocurrent", "source"]] \
        .dropna(subset=["sequence"]).query("sequence != ''")


def union_dedupe(d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([d2, d1], ignore_index=True)
    combined["ChR_name"] = combined["ChR_name"].astype(str).str.strip()
    combined = combined.drop_duplicates(subset=["ChR_name"], keep="first")
    combined = combined.drop_duplicates(subset=["sequence"], keep="first")
    return combined.reset_index(drop=True)


def load_parents_from_csvs(csv1: str, csv2: str) -> Dict[str, str]:
    parents: Dict[str, str] = {}
    for path in (csv1, csv2):
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        if "ChR_name" not in df.columns or "Amino_acid_sequence" not in df.columns:
            continue
        for canonical, aliases in PARENT_ALIASES.items():
            if canonical in parents:
                continue
            hit = df[df["ChR_name"].astype(str).str.strip().isin(aliases)]
            if not hit.empty:
                parents[canonical] = clean_seq(hit.iloc[0]["Amino_acid_sequence"])
    missing = set(PARENT_ALIASES) - set(parents)
    if missing:
        raise ValueError(f"Missing parent sequences for {missing}")
    return parents


@dataclass
class ReferenceStructure:
    sequence: str
    residue_numbers: List[int]
    ca_coords: np.ndarray
    contacts: List[Tuple[int, int]]


def fetch_pdb(pdb_id: str, cache_dir: Optional[Path] = None) -> str:
    pdb_id = pdb_id.upper()
    cache_dir = Path(cache_dir or ".pdb_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{pdb_id}.pdb"
    if f.exists():
        return f.read_text()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    f.write_text(r.text)
    return r.text


def parse_pdb_chain(pdb_text: str, chain: str = "A") -> Tuple[str, List[int], np.ndarray]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("ref", io.StringIO(pdb_text))
    seq: List[str] = []
    resnums: List[int] = []
    coords: List[np.ndarray] = []
    for model in structure:
        for ch in model:
            if ch.id != chain:
                continue
            for res in ch:
                hetflag, resnum, _ = res.id
                if hetflag.strip():
                    continue
                resname = res.get_resname()
                if resname not in THREE_TO_ONE or "CA" not in res:
                    continue
                seq.append(THREE_TO_ONE[resname])
                resnums.append(resnum)
                coords.append(res["CA"].get_coord())
        break
    if not seq:
        raise ValueError(f"No standard residues with CA in chain {chain}")
    return "".join(seq), resnums, np.array(coords, dtype=np.float32)


def build_contacts(ca: np.ndarray, cutoff: float = 4.5, min_seq_sep: int = 2) -> List[Tuple[int, int]]:
    L = ca.shape[0]
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    contacts: List[Tuple[int, int]] = []
    for i in range(L):
        for j in range(i + min_seq_sep, L):
            if d[i, j] <= cutoff:
                contacts.append((i, j))
    return contacts


def align_global(a: str, b: str) -> Tuple[str, str]:
    blosum62 = substitution_matrices.load("BLOSUM62")
    alns = pairwise2.align.globalds(a, b, blosum62, -11, -1, one_alignment_only=True)
    if not alns:
        return a, b
    aln = alns[0]
    return aln.seqA, aln.seqB


def project_positions(ref_seq: str, query_seq: str) -> Dict[int, Optional[int]]:
    a_ref, a_qry = align_global(ref_seq, query_seq)
    ref_pos = -1
    qry_pos = -1
    mapping: Dict[int, Optional[int]] = {}
    for cr, cq in zip(a_ref, a_qry):
        if cr != "-":
            ref_pos += 1
        if cq != "-":
            qry_pos += 1
        if cr != "-":
            mapping[ref_pos] = qry_pos if cq != "-" else None
    return mapping


@dataclass
class SchemaScorer:
    parents: Dict[str, str]
    pdb_id: str = "3UG9"
    chain: str = "A"
    contact_cutoff: float = 4.5
    min_seq_sep: int = 2
    cache_dir: Optional[Path] = None
    ref: ReferenceStructure = field(init=False)
    parent_at_ref: Dict[str, List[Optional[str]]] = field(init=False, default_factory=dict)

    def __post_init__(self):
        pdb_text = fetch_pdb(self.pdb_id, self.cache_dir)
        seq, resnums, ca = parse_pdb_chain(pdb_text, self.chain)
        contacts = build_contacts(ca, self.contact_cutoff, self.min_seq_sep)
        self.ref = ReferenceStructure(seq, resnums, ca, contacts)
        for name, pseq in self.parents.items():
            mapping = project_positions(self.ref.sequence, pseq)
            row: List[Optional[str]] = []
            for i in range(len(self.ref.sequence)):
                qi = mapping.get(i)
                row.append(pseq[qi] if qi is not None else None)
            self.parent_at_ref[name] = row

    def score(self, chimera_seq: str) -> Dict:
        chimera_seq = clean_seq(chimera_seq)
        mapping = project_positions(self.ref.sequence, chimera_seq)
        chim_at_ref: List[Optional[str]] = []
        for i in range(len(self.ref.sequence)):
            qi = mapping.get(i)
            chim_at_ref.append(chimera_seq[qi] if qi is not None else None)
        identity_sets: List[set] = []
        for i in range(len(self.ref.sequence)):
            c = chim_at_ref[i]
            if c is None:
                identity_sets.append(set())
                continue
            s = set()
            for pname, prow in self.parent_at_ref.items():
                if prow[i] == c:
                    s.add(pname)
            identity_sets.append(s)
        covered = sum(1 for s in identity_sets if s)
        coverage = covered / max(len(identity_sets), 1)
        E = 0
        evaluated = 0
        for (i, j) in self.ref.contacts:
            ci, cj = chim_at_ref[i], chim_at_ref[j]
            if ci is None or cj is None:
                continue
            evaluated += 1
            preserved = bool(identity_sets[i] & identity_sets[j])
            if not preserved:
                E += 1
        E_norm = E / evaluated if evaluated else float("nan")
        return {
            "E": E,
            "evaluated_contacts": evaluated,
            "total_contacts": len(self.ref.contacts),
            "E_norm": E_norm,
            "coverage": coverage,
        }

    def score_with_blocks(self, chimera_seq: str, parent_per_position: Sequence[str]) -> Dict:
        chimera_seq = clean_seq(chimera_seq)
        if len(parent_per_position) != len(chimera_seq):
            raise ValueError("parent_per_position length must equal chimera length")
        key_order = sorted(self.parents)
        chim_parent_at: List[str] = []
        for ch in parent_per_position:
            try:
                idx = int(ch)
            except ValueError:
                idx = -1
            chim_parent_at.append(key_order[idx] if 0 <= idx < len(key_order) else "")
        mapping = project_positions(self.ref.sequence, chimera_seq)
        chim_at_ref: List[Optional[str]] = [None] * len(self.ref.sequence)
        parent_at_ref: List[str] = [""] * len(self.ref.sequence)
        for ref_i, qi in mapping.items():
            if qi is None:
                continue
            chim_at_ref[ref_i] = chimera_seq[qi]
            parent_at_ref[ref_i] = chim_parent_at[qi]
        E = 0
        evaluated = 0
        for (i, j) in self.ref.contacts:
            ci, cj = chim_at_ref[i], chim_at_ref[j]
            pi, pj = parent_at_ref[i], parent_at_ref[j]
            if ci is None or cj is None or not pi or not pj:
                continue
            evaluated += 1
            preserved = False
            for pname, prow in self.parent_at_ref.items():
                if prow[i] == ci and prow[j] == cj:
                    preserved = True
                    break
            if not preserved:
                E += 1
        return {
            "E": E,
            "evaluated_contacts": evaluated,
            "total_contacts": len(self.ref.contacts),
            "E_norm": E / evaluated if evaluated else float("nan"),
        }


def enrich(
    df: pd.DataFrame,
    client: ESMClient,
    cache_root: Path,
    scorer: Optional[SchemaScorer] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    plddt_cache = cache_root / "plddt"
    emb_cache = cache_root / "emb"
    plddts: List[Optional[float]] = []
    embs: List[Optional[np.ndarray]] = []
    schema_E: List[Optional[int]] = []
    schema_En: List[Optional[float]] = []
    schema_cov: List[Optional[float]] = []
    for _, row in df.iterrows():
        p, _ = get_plddt(client, row["sequence"], plddt_cache)
        e = get_embedding(client, row["sequence"], emb_cache)
        plddts.append(p)
        embs.append(e)
        if scorer is not None:
            r = scorer.score(row["sequence"])
            schema_E.append(r["E"])
            schema_En.append(r["E_norm"])
            schema_cov.append(r["coverage"])
        else:
            schema_E.append(None)
            schema_En.append(None)
            schema_cov.append(None)
    df = df.copy()
    df["mean_plddt"] = plddts
    df["schema_E"] = schema_E
    df["schema_E_norm"] = schema_En
    df["schema_coverage"] = schema_cov
    keep = [i for i, e in enumerate(embs) if e is not None and plddts[i] is not None]
    df = df.iloc[keep].reset_index(drop=True)
    X = np.stack([embs[i] for i in keep], axis=0)
    return df, X
