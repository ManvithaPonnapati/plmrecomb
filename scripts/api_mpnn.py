from __future__ import annotations

import tempfile
from typing import Dict, List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from modal import App, Image, asgi_app, gpu


gpu_config = gpu.A10G(count=1)

app = App("plmrecomb-mpnn")
web_app = FastAPI()

image = (
    Image.debian_slim(python_version="3.11")
    .micromamba()
    .apt_install("wget", "git")
    .pip_install("git+https://github.com/sokrypton/ColabDesign.git@v1.1.1")
    .pip_install("google-cloud-storage", "google-auth")
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu=gpu_config,
    )
)

with image.imports():
    from colabdesign.mpnn import mk_mpnn_model


@app.function(image=image, gpu=gpu_config, concurrency_limit=5)
def sample(pdb_string: str, mpnn_config: dict) -> List[Dict]:
    model = mk_mpnn_model()
    fix_pos = str(mpnn_config.get("fix_pos", ""))
    inverse = bool(mpnn_config.get("inverse", False))
    sampling_temp = float(mpnn_config.get("temp", 0.1))
    batch = int(mpnn_config.get("batch", 8))
    chains = mpnn_config.get("chains", "A")
    with tempfile.NamedTemporaryFile(delete=True, mode="w+", suffix=".pdb") as tf:
        tf.write(pdb_string)
        tf.flush()
        model.prep_inputs(
            pdb_filename=tf.name,
            chain=chains,
            inverse=inverse,
            fix_pos=fix_pos,
        )
    out = model.sample_parallel(temperature=sampling_temp, batch=batch)
    return [
        {"score": float(out["score"][n]),
         "seqid": float(out["seqid"][n]),
         "seq": out["seq"][n]}
        for n in range(batch)
    ]


@app.function(image=image, gpu=gpu_config, concurrency_limit=5)
def score(pdb_string: str, sequence: str, chains: str = "A") -> Dict:
    model = mk_mpnn_model()
    with tempfile.NamedTemporaryFile(delete=True, mode="w+", suffix=".pdb") as tf:
        tf.write(pdb_string)
        tf.flush()
        model.prep_inputs(pdb_filename=tf.name, chain=chains)
    out = model.score(sequence=sequence)
    return {"score": float(out["score"]), "seq": sequence}


@app.function(image=image)
def generate_sequences(pdb_string: str, mpnn_config: dict):
    return sample.remote(pdb_string, mpnn_config)


@web_app.post("/sample")
async def sample_endpoint(json_data: dict):
    return JSONResponse(content=await sample.remote.aio(
        json_data["pdb_string"], json_data["params"]
    ))


@web_app.post("/score")
async def score_endpoint(json_data: dict):
    return JSONResponse(content=await score.remote.aio(
        json_data["pdb_string"],
        json_data["sequence"],
        json_data.get("chains", "A"),
    ))


@web_app.get("/")
async def root() -> dict:
    return {"service": "plmrecomb-mpnn",
            "endpoints": ["POST /sample", "POST /score"]}


@app.function()
@asgi_app()
def fastapi_app():
    return web_app
