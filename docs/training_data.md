# External training labels and augmentation

| Source | Train | Validation | Test | Target provenance |
|---|---:|---:|---:|---|
| DFL public release | 5 | 1 | 1 | Provider ball-status and team-possession fields |
| SkillCorner open data | 16 | 2 | 2 | Provider in-play phases and possession team; complement within match periods is dead |
| Metrica sample data | 3 | 0 | 0 | Continuous H/A/D states reconstructed by our code from provider team-event logs |

All splits are by whole match, seed 42. The three Metrica matches are event-derived supervision, not supplied frame-level ground truth. No human-labelled or algorithmically labelled Barcelona examples were used to train these models. The technical document describes target mapping and uncertain transition timing.

Sources: [DFL Figshare](https://figshare.com/articles/dataset/28196177), [SkillCorner](https://github.com/SkillCorner/opendata), [Metrica](https://github.com/metrica-sports/sample-data). The download manifest records the source URLs and checksums. Provider labeler identities are not specified in the retained source metadata; we did not create new manual labels.

Both models use the same source-specific augmentation. DFL dead-ball masking hides 90% of observed dead-play samples with gap/burst placement and approximately 5.8% fully missing all-dead windows. DFL ordinary live masking retains the earlier 15% run probability with 20–60% contiguous hiding. SkillCorner and Metrica retain natural observations without extra ordinary dead-ball hiding. SkillCorner keeps finite detected-only coordinates. All three sources keep the original live goalkeeper-control masking (at least 75% of observed candidate tokens), with targets unchanged. No player dropout or camera cropping is used; validation and test data are not augmented.

Automatic target-game visibility statistics informed the DFL policy; they are not possession training targets. Published training configs preserve the measured visibility fraction and profile hash. No target-game feature cache or derived tracking is distributed.
