# Reproducibility contract toolkit

This toolkit initializes and audits the executable research contract that must
exist before Figure 2 is redesigned as a reproducibility workflow.

All filenames, labels, headers, and documentation are in English.

## Step 1 — initialize the draft contract

```bash
python3 reproducibility_contract_v1_en/bootstrap_reproducibility_contract.py \
  --root . \
  --source-workspace <A3_LASRC_WORKSPACE>
```

The draft is created under:

```text
local_only/workflow_contract_draft/
```

It remains outside public Zenodo packages while incomplete.

## Step 2 — inspect the audit

```bash
cat \
  local_only/workflow_contract_draft/docs/provenance/WORKFLOW_REPRODUCIBILITY_AUDIT.md
```

The expected initial status is `INCOMPLETE`. This is not an execution failure;
it identifies the exact evidence still required.

## Optional — scan for candidate evidence

```bash
python3 reproducibility_contract_v1_en/scan_reproducibility_candidates.py \
  --workspace <A3_LASRC_WORKSPACE> \
  --out local_only/workflow_contract_draft/metadata/candidate_inventory.tsv
```

## Re-run the audit after completing manifests

```bash
python3 reproducibility_contract_v1_en/audit_reproducibility_contract.py \
  --root .
```

## Promotion

Do not promote the contract until the audit status is
`READY_FOR_PROMOTION`.

```bash
python3 reproducibility_contract_v1_en/promote_reproducibility_contract.py \
  --root .
```

The promotion script refuses to run while the contract is incomplete.
