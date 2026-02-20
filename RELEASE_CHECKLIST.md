# Release Checklist (Phase 6)

## 1. Test and Build
- [ ] `python -m compileall core memory scripts main.py ui/dashboard.py`
- [ ] `python scripts/load_test.py --requests 24 --workers 6 --listener-blocks 20`
- [ ] `python scripts/backfill_jobs.py --batch-size 25 --max-batches 20`
- [ ] Smoke check: `streamlit run ui/dashboard.py`

## 2. Backend Health
- [ ] Graph backend status verified (Neo4j or JSON fallback)
- [ ] Vector backend status verified (Qdrant or local fallback)
- [ ] Provenance coverage >= 0.95
- [ ] Drift alarms reviewed

## 3. Versioning and Migration
- [ ] Active ISA version confirmed in `spec/isa_registry.json`
- [ ] Migration dry-run done with `scripts/isa_migrate.py`
- [ ] Roll-forward/rollback migration paths documented

## 4. Rollback Plan
- [ ] Snapshot graph (`memory/global_graph.json` or Neo4j backup)
- [ ] Snapshot vector index (`memory/vector_index.jsonl` or Qdrant snapshot)
- [ ] Keep previous release tag and commit hash
- [ ] Validate rollback command path before deploy

## 5. Release Decision
- [ ] Load test report acceptable
- [ ] Backfill report acceptable
- [ ] No blocking errors in dashboard or listener logs
- [ ] Release approved
