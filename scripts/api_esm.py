from __future__ import annotations

import os
from io import StringIO
from typing import Dict, List

import modal
from fastapi import FastAPI
from fastapi.responses import JSONResponse


app = modal.App("plmrecomb-esm")
web_app = FastAPI()

GPU_TYPE = os.environ.get("ESM_GPU", "A10G")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.3.1",
        "transformers==4.49.0",
        "numpy",
        "biotite==0.41.2",
        "fastapi",
        "uvicorn",
    )
)


@app.cls(image=image, gpu=GPU_TYPE, timeout=60 * 60, container_idle_timeout=300)
class ESMService:
    def __init__(self):
        self.fold_model = None
        self.fold_tok = None
        self.emb_model = None
        self.emb_tok = None
        self.device = "cuda"

    @modal.enter()
    def load(self):
        import torch
        from transformers import EsmForProteinFolding, AutoTokenizer, AutoModel

        fold_id = "facebook/esmfold_v1"
        self.fold_tok = AutoTokenizer.from_pretrained(fold_id)
        self.fold_model = EsmForProteinFolding.from_pretrained(fold_id).to(self.device).eval()
        self.fold_model.trunk.set_chunk_size(64)

        emb_id = "facebook/esm2_t33_650M_UR50D"
        self.emb_tok = AutoTokenizer.from_pretrained(emb_id)
        self.emb_model = AutoModel.from_pretrained(emb_id).to(self.device).eval()

    @modal.method()
    def fold_to_pdb(self, sequence: str) -> Dict:
        if not isinstance(sequence, str) or not sequence.strip():
            return {"success": False, "error": "'sequence' must be a non-empty string"}
        seq = sequence.strip().replace(" ", "").replace("-", "")
        if any(c in seq for c in ":*|>\n\t\r"):
            return {"success": False, "error": "Provide a raw amino-acid sequence."}
        try:
            import torch
            import numpy as np
            from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
            from transformers.models.esm.openfold_utils.residue_constants import (
                atom_types, restypes, restype_1to3,
            )
            from biotite.structure import Atom, array
            from biotite.structure.io.pdb import PDBFile, set_structure

            toks = self.fold_tok(
                [seq], return_tensors="pt", add_special_tokens=False,
                truncation=True, padding=False, max_length=1024,
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.inference_mode():
                out = self.fold_model(**toks)

            positions37 = atom14_to_atom37(out["positions"][-1], out)
            exists37 = out["atom37_atom_exists"].bool()
            aatype = out["aatype"][0].cpu().numpy()
            plddt = out["plddt"][0].cpu().numpy()
            residue_index = out["residue_index"][0].cpu().numpy().tolist()

            L = positions37.shape[1]
            atoms = []
            ca_plddts: List[float] = []
            for i in range(L):
                aa_1 = restypes[aatype[i]]
                resname = restype_1to3[aa_1]
                res_id = int(residue_index[i])
                for a_idx in range(37):
                    if not bool(exists37[0, i, a_idx]):
                        continue
                    xyz = positions37[0, i, a_idx].detach().cpu().numpy()
                    aname = atom_types[a_idx]
                    elem = aname[0]
                    b = float(plddt[i, a_idx])
                    if aname == "CA":
                        ca_plddts.append(b)
                    atoms.append(Atom(
                        coord=xyz, chain_id="A", atom_name=aname,
                        res_name=resname, res_id=res_id, element=elem, b_factor=b,
                    ))

            pdb_file = PDBFile()
            set_structure(pdb_file, array(atoms))
            sio = StringIO()
            pdb_file.write(sio)
            mean_plddt = float(np.mean(ca_plddts)) if ca_plddts else float("nan")
            return {"success": True, "pdb_str": sio.getvalue(),
                    "mean_plddt": mean_plddt, "length": L}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @modal.method()
    def embed(self, sequence: str) -> Dict:
        if not isinstance(sequence, str) or not sequence.strip():
            return {"success": False, "error": "'sequence' must be a non-empty string"}
        seq = sequence.strip().replace(" ", "").replace("-", "")
        if any(c in seq for c in ":*|>\n\t\r"):
            return {"success": False, "error": "Provide a raw amino-acid sequence."}
        try:
            import torch
            toks = self.emb_tok(
                [seq], return_tensors="pt", add_special_tokens=True,
                truncation=True, padding=False, max_length=1024,
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.inference_mode():
                out = self.emb_model(**toks)
            hidden = out.last_hidden_state[0]
            attn = toks["attention_mask"][0].bool()
            hidden = hidden[1:-1]
            attn = attn[1:-1]
            hidden = hidden[attn]
            emb = hidden.mean(dim=0).cpu().numpy().tolist()
            return {"success": True, "embedding": emb, "dim": len(emb),
                    "length": int(attn.sum().item())}
        except Exception as e:
            return {"success": False, "error": str(e)}


@app.function(image=image, gpu=GPU_TYPE)
@modal.asgi_app()
def startapp() -> FastAPI:
    service = ESMService()

    @web_app.post("/predict")
    async def predict(json_data: Dict):
        seq = json_data.get("sequence")
        try:
            result = await service.fold_to_pdb.remote.aio(sequence=seq)
        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
        return JSONResponse(status_code=200 if result.get("success") else 400, content=result)

    @web_app.post("/embed")
    async def embed(json_data: Dict):
        seq = json_data.get("sequence")
        try:
            result = await service.embed.remote.aio(sequence=seq)
        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
        return JSONResponse(status_code=200 if result.get("success") else 400, content=result)

    @web_app.get("/")
    async def root():
        return {"service": "plmrecomb-esm",
                "endpoints": ["POST /predict", "POST /embed"], "gpu": GPU_TYPE}

    return web_app
