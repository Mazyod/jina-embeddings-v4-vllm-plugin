"""Revalidation toolkit — run after a vLLM upgrade to confirm the plugin still fits the API.

The JinaV4MultiVector model class subclasses Qwen2.5-VL and uses vLLM-internal pooler builders, so a
vLLM bump can move the APIs it depends on. These cheap (mostly CPU) jobs regenerate the exact facts
the model class relies on, plus a GPU smoke that the pooling path still returns per-token vectors:

    uv run modal run src/jinav4_vllm/modal_app/revalidate.py::verify_projector       # GPU
    uv run modal run src/jinav4_vllm/modal_app/revalidate.py::recon_qwen25_api       # CPU
    uv run modal run src/jinav4_vllm/modal_app/revalidate.py::recon_dump_sources     # CPU
    uv run modal run src/jinav4_vllm/modal_app/revalidate.py::recon_vllm_pooling     # CPU
    uv run modal run src/jinav4_vllm/modal_app/revalidate.py::inspect_pooler         # GPU
    uv run modal run src/jinav4_vllm/modal_app/revalidate.py::spike_offline_shape    # GPU

See docs/COMPAT.md for the version table and the upgrade checklist.
"""
from __future__ import annotations

from jinav4_vllm.modal_app.app import app, ref_image, vllm_image, GPU, COMMON, ART


@app.function(image=ref_image, gpu=GPU, timeout=1800, **COMMON)
def verify_projector():
    """Our extracted W,b must reproduce the model's own retrieval projector (max_abs_diff < 1e-3)."""
    import sys; sys.path.insert(0, "/root")
    import numpy as np, torch
    from transformers import AutoModel
    proj = np.load(f"{ART}/projector/retrieval.npz")
    W, b = proj["W"], proj["b"]
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True,
                                      torch_dtype=torch.float32).eval()
    for setter in ("set_adapter", "set_task"):
        if hasattr(model, setter):
            try: getattr(model, setter)("retrieval")
            except Exception: pass
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.standard_normal((16, 2048)), dtype=torch.float32)
    with torch.no_grad():
        ref = model.multi_vector_projector(x).cpu().numpy()
    ours = x.numpy() @ W.T + b
    mad = float(np.abs(ref - ours).max())
    print(f"max_abs_diff={mad:.6e}", "PASS" if mad < 1e-3 else "FAIL")
    assert mad < 1e-3, "extracted projector does not match model projector"
    return mad


@app.function(image=vllm_image, timeout=1800, **COMMON)
def recon_vllm_pooling():
    """CPU recon: dump the vLLM pooling API surface needed for an in-engine projected pooler.

    Dumps: registered architectures (Col*/embed/Qwen-VL), the pooling-adapter source (how --convert
    embed attaches a pooler), the pooler module surface, and an existing projected-multivector model
    (ColQwen/ColPali) to copy the pattern.
    """
    import inspect, os, importlib
    out = []
    def dump(title, s, cap=6000):
        out.append(f"\n===== {title} =====\n{s[:cap]}")

    try:
        from vllm import ModelRegistry
        archs = sorted(ModelRegistry.get_supported_archs())
        interesting = [a for a in archs if any(k in a.lower() for k in
                       ("col", "embed", "qwen2_5_vl", "qwen2vl", "jina", "gme", "llavanext", "vlm2vec"))]
        dump("ARCHS (interesting)", "\n".join(interesting))
    except Exception as e:
        dump("ARCHS error", repr(e))

    for mod in ("vllm.model_executor.models.adapters",):
        try:
            m = importlib.import_module(mod)
            names = [n for n in dir(m) if not n.startswith("_")]
            dump(f"{mod} names", ", ".join(names))
            for fn in ("as_embedding_model", "as_seq_cls_model", "_create_pooling_model_cls", "as_reward_model"):
                if hasattr(m, fn):
                    try: dump(f"{mod}.{fn}", inspect.getsource(getattr(m, fn)))
                    except Exception as e: dump(f"{mod}.{fn} src err", repr(e))
        except Exception as e:
            dump(f"{mod} err", repr(e))

    try:
        import vllm.model_executor.layers.pooler as P
        dump("pooler public names", ", ".join(n for n in dir(P) if not n.startswith("_")))
        for cls in ("Pooler", "DispatchPooler", "PoolerHead", "PoolingType", "PoolerOutput",
                    "PoolingParamsUpdate", "PoolingMetadata", "build_output", "AllPool", "EmbeddingPoolerHead"):
            obj = getattr(P, cls, None)
            if obj is None: continue
            try:
                if inspect.isclass(obj):
                    meths = [n for n in dir(obj) if not n.startswith("_")]
                    sig = ""
                    for mm in ("__init__", "forward", "for_encode", "for_embed", "get_pooling_updates"):
                        f = getattr(obj, mm, None)
                        if f:
                            try: sig += f"\n  {mm}{inspect.signature(f)}"
                            except Exception: pass
                    dump(f"pooler.{cls}", f"methods={meths}\nsigs={sig}")
                else:
                    dump(f"pooler.{cls}", repr(obj))
            except Exception as e:
                dump(f"pooler.{cls} err", repr(e))
    except Exception as e:
        dump("pooler import err", repr(e))

    try:
        import vllm.model_executor.models as M
        pkgdir = os.path.dirname(M.__file__)
        cand = [f for f in os.listdir(pkgdir) if any(k in f.lower() for k in ("col", "jina", "gme"))]
        dump("model files (col/jina/gme)", ", ".join(sorted(cand)))
        for f in cand:
            if f.endswith(".py"):
                src = open(os.path.join(pkgdir, f)).read()
                lines = src.splitlines()
                hits = [i for i, l in enumerate(lines) if any(k in l for k in
                        ("pooler", "Pooler", "token_embed", "projector", "projection", "multi_vector", "nn.Linear", "normalize"))]
                snippet = "\n".join(f"{i:4d}: {lines[i]}" for i in hits[:60])
                dump(f"model {f} (pooler/projection lines)", snippet, cap=4000)
    except Exception as e:
        dump("models scan err", repr(e))

    text = "".join(out)
    print(text)
    return {"len": len(text)}


@app.function(image=vllm_image, timeout=1800, **COMMON)
def recon_dump_sources():
    """Dump full source of the closest template model + the token-embed pooler builder."""
    import os
    import vllm.model_executor.models as M
    pkg = os.path.dirname(M.__file__)
    out = []
    def cat(path, title, cap=20000):
        try:
            out.append(f"\n########## {title} ({path}) ##########\n" + open(path).read()[:cap])
        except Exception as e:
            out.append(f"\n########## {title} ERR {e!r} ##########")
    cat(os.path.join(pkg, "colqwen3.py"), "colqwen3.py")
    import vllm.model_executor.layers.pooler.tokwise as TW
    cat(TW.__file__, "pooler/tokwise.py")
    cat(os.path.join(pkg, "colpali.py"), "colpali.py", cap=14000)
    text = "".join(out)
    print(text)
    return {"len": len(text)}


@app.function(image=vllm_image, timeout=1800, **COMMON)
def recon_qwen25_api():
    """Confirm Qwen2.5-VL class names + import paths the JinaV4MultiVector plugin depends on."""
    import inspect, importlib
    out = {}
    m = importlib.import_module("vllm.model_executor.models.qwen2_5_vl")
    out["qwen2_5_vl names"] = [n for n in dir(m) if "Qwen2_5_VL" in n]
    for path, names in {
        "vllm.model_executor.layers.pooler.tokwise": ["pooler_for_token_embed"],
        "vllm.model_executor.models.interfaces": ["SupportsLateInteraction", "SupportsMultiModal"],
        "vllm.model_executor.models.interfaces_base": ["default_pooling_type"],
        "vllm.model_executor.models.utils": ["AutoWeightsLoader", "WeightsMapper"],
        "vllm.model_executor.model_loader.weight_utils": ["default_weight_loader"],
        "vllm.multimodal": ["MULTIMODAL_REGISTRY"],
    }.items():
        try:
            mod = importlib.import_module(path)
            out[path] = {n: hasattr(mod, n) for n in names}
        except Exception as e:
            out[path] = f"ERR {e!r}"
    try:
        from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_embed
        out["pooler_for_token_embed sig"] = str(inspect.signature(pooler_for_token_embed))
    except Exception as e:
        out["pooler_for_token_embed sig"] = f"ERR {e!r}"
    try:
        from vllm.config import ModelConfig
        out["ModelConfig has head_dtype"] = "head_dtype" in dir(ModelConfig)
    except Exception as e:
        out["ModelConfig head_dtype"] = f"ERR {e!r}"
    import json; print(json.dumps(out, indent=2, default=str))
    return out


@app.function(image=vllm_image, gpu=GPU, timeout=1800, **COMMON)
def inspect_pooler():
    """GPU recon: learn the live model class + pooler interface for an in-engine projection."""
    import inspect
    from vllm import LLM
    from vllm.config import PoolerConfig
    llm = LLM(model="jinaai/jina-embeddings-v4-vllm-retrieval", runner="pooling",
              pooler_config=PoolerConfig(task="token_embed"), max_model_len=512,
              gpu_memory_utilization=0.5, enforce_eager=True)
    info = {}
    try:
        runner = llm.llm_engine.model_executor.driver_worker.model_runner
        model = runner.model
    except Exception as e:
        info["driver_path_error"] = repr(e)
        model = None
        for attr in ("llm_engine", "engine"):
            eng = getattr(llm, attr, None)
            if eng is not None:
                info["engine_attrs"] = [a for a in dir(eng) if not a.startswith("__")][:40]
                break
    if model is not None:
        info["model_class"] = type(model).__module__ + "." + type(model).__qualname__
        info["model_mro"] = [c.__module__ + "." + c.__name__ for c in type(model).__mro__]
        info["has_pooler"] = hasattr(model, "pooler")
        if hasattr(model, "pooler"):
            p = model.pooler
            info["pooler_class"] = type(p).__module__ + "." + type(p).__qualname__
            info["pooler_callables"] = [a for a in dir(p) if not a.startswith("_")][:40]
            try:
                info["pooler_forward_sig"] = str(inspect.signature(p.forward))
            except Exception as e:
                info["pooler_forward_sig_err"] = repr(e)
    print(info)
    return info


@app.function(image=vllm_image, gpu=GPU, timeout=1800, **COMMON)
def spike_offline_shape():
    """Confirm offline pooling returns [n, 2048] per-token hidden states + matching token ids."""
    from vllm import LLM
    from vllm.config import PoolerConfig
    llm = LLM(model="jinaai/jina-embeddings-v4-vllm-retrieval",
              runner="pooling", pooler_config=PoolerConfig(task="token_embed"),
              max_model_len=1024, gpu_memory_utilization=0.8)
    out = llm.encode(["Query: hello world"])[0]
    data = out.outputs.data
    shape = tuple(data.shape)
    ntok = len(out.prompt_token_ids)
    print(f"data.shape={shape} n_prompt_tokens={ntok}")
    assert shape[1] == 2048, f"expected hidden dim 2048, got {shape[1]}"
    assert shape[0] == ntok, "row count must equal prompt token count"
    return {"shape": shape, "n_tokens": ntok}
