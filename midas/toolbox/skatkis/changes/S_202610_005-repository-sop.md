# S_202610_005: Repository SOP and README

| Field | Value |
|---|---|
| Status | ready |
| Priority | low |
| Kind | documentation |
| Depends on | S_202610_001 |
| Raised | 2026-10-31 |

## Description

The example repository has an engineering SOP (`SOP_Modular_Analytical_Tools.md`) and
a repository README that the agent rules refer to by section. This repository has the
value-investing spec but no general engineering standard and no README, so
`AGENTS.md` and `.claude/rules/` carry the engineering rules for now.

## Approach

Adapt the example SOP's tool-agnostic chapters (philosophy, layout, file ownership,
schema, sources, registry and hydration, contracts, gates, registries, function design,
documentation, validation, testing, orchestration, AI-assisted development) into
`Toolbox/SOP_Modular_Analytical_Tools.md`, dropping what does not apply (client
folders, accreditation sign-off, Word reports). Write a repository `README.md`
(purpose, layout, setup, commands, the Midas server). Then point `AGENTS.md` and the
rules at SOP sections.

## Done when

- The SOP and README exist, and the rules cite SOP sections instead of carrying the full
  text.
