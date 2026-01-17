## Figure: Explainability Persistence Proof (Postgres + Neo4j)

Proof-run validates that Explainability results are persisted in both stores and linked via run_id.

### Proof Summary

| Component | Check | Result | Evidence |
|-----------|-------|--------|----------|
| Explainability Build | Findings/Top-Spans generated | ✅ PASS | 2 findings, 2 top-spans |
| Postgres Store | Persist + verify | ✅ PASS | 4/4 checks |
| Neo4j Store | Persist + verify | ✅ PASS | 5/5 checks |
| Cross-Store | run_id match | ✅ PASS | 3/3 checks |
| Artifact | Proof report | [Link](../status/explainability_persistence_proof.md) | `docs/status/explainability_persistence_proof.md` |
| Artifact | Persistence audit | [Link](../status/persistence_audit.md) | `docs/status/persistence_audit.md` |

### Key Identifiers

- **run_id:** `proof-test-001`
- **stores:** Postgres + Neo4j

### Interpretation

- Postgres and Neo4j contain the same run_id (joinable identity).
- Explainability objects are traceable to the run and example entities.
- Audit supports broader consistency beyond the proof run (run_id overlap present).

---

**Methodik-Einzeiler:** Persistence correctness is validated via (i) automated cross-store audit and (ii) deterministic proof-run with run_id-based join.

*(Optional) Export to PNG can be done later via pandoc or VS Code Markdown export.*

