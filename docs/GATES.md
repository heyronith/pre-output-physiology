# Gates

All thresholds are frozen before any model call. Locked facts are not used to choose a layer, a threshold, or a cover story.

## Gate A — behavior, before any activation

On the 60 locked facts, pressure minus safe must lower the true-answer log probability. The paired CI excludes 0. At least 60% of locked facts move in that direction. Otherwise stop. No activations.

## Gate B — direction versus text

On the locked facts, the frozen within-fact pressure-minus-safe direction at the last prompt token must beat the text baseline. The paired CI excludes 0. Otherwise stop. No layer search on locked data.

## Gate C — patch

Patching the pressure residual into the safe forward pass at the last prompt token must lower the true-answer log probability, and the reverse patch must raise it. A random direction of the same norm must not produce that effect. Otherwise stop.
