# Pre-token concealment direction

**Status: Step 2 protocol frozen. No data collected.**

The previous project, phases 1–28, is frozen on branch `archive/preoutput-v1` and tag `archive-preoutput-v1`, with the original phase branches left in place. Its files remain under `archive/v1/` for inspection. Those results are not evidence for this protocol. See `docs/ARCHIVE.md`.

## Claim to be tested

On paired prompts that share a known fact, a pressure context can lower the log probability of the true answer before any output token is sampled. The residual stream at the last prompt token may carry a within-fact pressure-minus-safe direction that beats a text baseline and, when patched, moves that log probability. A random direction of the same norm must not.

## Frozen choices

- Primary model, not yet run: `meta-llama/Llama-3.1-8B-Instruct`
- Replication model, not authorized: `Qwen/Qwen2.5-14B-Instruct`
- Position, if activations are ever collected: last prompt token only
- Development facts: 120. Locked facts: 60. Locked cover stories disjoint from development.
- GPU is forbidden until a later approved step authorizes Modal.

## Stop rules

Stop if pressure does not lower the true-answer log probability on locked facts. Stop if a text classifier matches the activation direction. Stop if patching does nothing or the random-direction control also works. No layer search on locked data.

