# zeno-rp Agent Instructions

## Mandatory startup

- Before proposing or changing anything related to LP-ACT, mobile manipulation action representations, ACT action chunks, waypoint labels, or execution timing, read `docs/LP_ACT_V1_HANDOFF.md` in full.
- Treat that document as the current shared research state. Do not restart the discussion from generic waypoint-policy ideas.
- Then inspect the actual repository and data schema. Clearly separate facts verified in code/data from assumptions recorded in the handoff.

## Working style

- Communicate with the user primarily in Chinese; keep established English technical terms where they are clearer.
- Explain one conceptual layer at a time and define every symbol operationally.
- When the user challenges an assumption, answer the precise objection before extending the method.
- Prefer the smallest testable implementation. The immediate goal is LP-ACT V1, not a more ambitious learned phase controller.
- Do not claim novelty before checking the closest work and isolating a falsifiable advantage.
- Never silently reinterpret a commanded velocity integral as ground-truth physical motion.

## Engineering rules

- Start with read-only inspection and a modification plan unless the user explicitly asks for implementation.
- Preserve the existing ACT backbone for V1; change the label generator, output parameterization, loss/output dimensions, and execution adapter only where necessary.
- Before full training, require: timestamp/schema audit, SE(2) convention tests, label visualizations, action-to-waypoint-to-action reconstruction, and a tiny-set overfit test.
- Do not run robot hardware, delete data, overwrite checkpoints, or change system configuration without explicit confirmation.
- Keep new experiment configs and outputs uniquely named. Record exact commands and parameter values.

## Updating shared context

- After a material research decision, update `docs/LP_ACT_V1_HANDOFF.md` rather than relying on chat history.
- Record: the decision, why it was made, evidence, rejected alternatives, unresolved questions, and the next concrete experiment.
