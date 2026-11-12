# Diagnostic-only route. Never installed into production server.py.
@app.post('/__r144_quality')
async def r144_quality():
    from pathlib import Path
    from prefill_nll import score_model
    p=Path('/models/fnx-diagnostics/r144-raw-norm-20261112')
    eng=get_engine()
    def work():
        with LOCK,torch.no_grad():
            eng.m.id=uuid.uuid4().hex
            fixtures=json.loads((p/'fixtures.json').read_text())
            return {name:score_model(eng.m,ids) for name,ids in fixtures.items()}
    async with REQUEST_LOCK:return await asyncio.to_thread(work)
