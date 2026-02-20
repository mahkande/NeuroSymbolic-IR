# ISA Migration Policy

## Supported Path
- `1.0 -> 1.1`

## Rules
- `ATTR` key normalization:
  - `özellik` or `property` -> `ozellik`
- `BELIEVE` confidence normalization:
  - Clamp to `[0.00, 1.00]`
  - Format as string with 2 decimals

## Safety
- Unknown opcodes are preserved (no destructive rewrite).
- Argument count is preserved; structural changes are not applied automatically.
- Validation remains enforced by `CognitiveValidator` after migration.

## Tooling
- Registry: `spec/isa_registry.json`
- Migration CLI: `scripts/isa_migrate.py`
