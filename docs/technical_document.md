# Learning Football Possession When the Ball Disappears

**Submission technical document · 17 November 2026**  
**Team and authors:** Pending team details

## 1 Why a Transformer

Possession is a temporal state. A ball passing between teammates can be far from every player without changing ownership. An opponent can approach or touch it without establishing control. A goalkeeper can hold it while the tracker loses sight of it. We wanted a model that could use the movement before and after an ambiguous moment rather than decide possession from a single nearest-player measurement.

That led us to a temporal Transformer. The input is a short sequence of player and ball observations, and the output is home possession, away possession, or dead ball at its centre. Attention lets the centre prediction use relevant moments elsewhere in the window. This is an offline task: future observations are available, so a sequence can contain evidence of a pass arriving, a restart beginning, or the ball returning after an occlusion.

Our original discussion considered ±7.5 seconds of context and one prediction every half second. Subsequent experiments moved to ±5 seconds with predictions every quarter second. Our submitted system uses two windows: ±5 seconds (51 observations) and ±10 seconds (101 observations), both sampled at 5 Hz. The shorter window retains responsive team predictions; the longer window supplies a second estimate of whether play is dead. We do not claim that a Transformer is necessary or optimal; it is a practical way to learn temporal evidence beyond hand-written proximity rules.

The other architectural requirement was player-order invariance. A tracking slot is not a meaningful rank, and substitutions and identity switches make it a poor identity signal. The completed baseline applies the same encoder to every player, then aggregates players within each team. Player positions matter; their ordering in the input does not.

## 2 Recovering the ball from the second camera feed

Before asking a model to reason through missing observations, we tried to recover observations already present in the supplied data. The tactical and broadcast recordings have different timelines and observation patterns. A ball missing from tactical tracking can still be observed in the broadcast feed, but the feeds cannot be joined by matching frame numbers directly.

We aligned them using team-labelled player geometry. Coarse signatures of the teams' spatial distributions supplied candidate correspondences, followed by local offset refinement and frame-level geometric checks. The implemented mapping is:

```text
first half:  tactical frame = broadcast frame + 4,014
second half: tactical frame = broadcast frame + 26,920
```

These offsets correspond to 160.56 and 1,076.8 seconds at 25 Hz. We did not use ball positions or ground-truth possession labels to estimate them. An accepted correspondence requires valid broadcast homography, at least 18 close directional player matches, and a 65th-percentile symmetric nearest-player distance no larger than 2.5 m. These checks help reject close-ups and replays that do not agree with the tactical player geometry.

We retain an existing tactical ball observation whenever one exists. Geometry-accepted broadcast ball observations fill tactical gaps only; they do not overwrite tactical observations. The original supplement retained 4,346 in-bounds frames. The revised version also retains 884 out-of-bounds observations, bringing the total to 5,230 frames, or 209.2 seconds. Those outside-pitch observations can supply evidence during stoppages that the original in-bounds-only filter discarded. Where both feeds observe the ball, their median position disagreement is approximately 0.75 m, with a 90th percentile of 5.47 m. Alignment and ball tracking remain approximate.

This supplement is important twice: it improves the inputs used for Barcelona inference, and it gives the rule-based comparison labels more evidence before they decide that the ball is missing. It does not supply training examples from Barcelona. The external training matches use their own observed tracking. The expanded models keep the broadcast-source feature column fixed to zero, including at inference, rather than activating an untrained source indicator.

## 3 Building a comparison reference and finding the dead-ball failure

Our early rules gave us an interpretable reference for reviewing the game. The aim was to retain a team's possession through passes and challenges until there was sustained evidence of control by the other side, rather than continually assign the ball to whichever player happened to be nearest.

For an observed ball, a team provides control evidence when its nearest player is within 2 m and at least 0.75 m closer than the nearest opponent, with observed players from both teams available. Initial control requires ten consecutive frames, or 0.4 seconds. A confirmed takeover requires 20 frames, or 0.8 seconds, and is backfilled to the start of the confirming streak. Ambiguity and short missing-ball gaps retain the established owner. State resets between halves.

The first dead-ball assumptions were deliberately simple: an observed ball outside the pitch is dead; a missing-ball run of at least five seconds is marked dead retrospectively from its beginning. Reviewing these cases exposed the key failure. A ball can disappear because play has stopped, but it can also disappear because a goalkeeper is holding it. Missingness is a property of the observation process, not a reliable definition of whether play is live.

We improved the comparison reference with goalkeeper-control logic. Candidate exceptions require evidence linking the missing ball to the same tracked goalkeeper: repeated close control or a plausible incoming trajectory, continuity and visibility of the goalkeeper in the penalty area, and plausible recovery or distribution evidence after the gap. The frozen version accepts six goalkeeper-control intervals. We also investigated goalkeeper team inconsistencies. For submission preprocessing, defending ends are inferred independently in each half from a majority of provider-labelled goalkeeper observations near either goal. Goalkeeper slots observed at the opposing end receive the inferred defending team. No manually chosen orientation or frame-specific correction table is used. The possession comparison reference remains separate from this preprocessing.

Automatic set-piece tags provide another source of evidence. At a tag's start, the reference marks dead play for a three-second minimum hold, then releases at the first observed ball before the tag ends; the tag end caps the hold if no qualifying observation occurs, including when the tag is shorter than three seconds. Normal control rules resume after release. This describes the implemented assumption, not guaranteed whistle or restart timing.

These outputs are **pseudo-labels for comparing model behaviour**, not verified ground truth. They allow us to inspect where models disagree with our possession assumptions and with one another. They are not training targets for the final external-only models. The public interval, frames 11,185–18,684, remains excluded from this comparison reference and from the evaluation contexts. Any match-specific reference rule must be distinguished from a general learned model improvement.

## 4 Getting training labels from the external datasets

We retained three external sources because they offer different routes to supervised possession targets. The challenge was to map their annotation semantics and tracking formats into a common representation without treating missing tracking as a label.

| Source | Training | Validation | Test | How we obtain H, A, and D |
|---|---:|---:|---:|---|
| DFL | 5 matches | 1 match | 1 match | Provider ball-status and team-possession fields |
| SkillCorner | 16 matches | 2 matches | 2 matches | Provider in-play phases and possession team, bounded by official match periods |
| Metrica | 3 matches | — | — | A continuous state reconstructed from provider team-event logs |
| **Total** | **24 matches** | **3 matches** | **3 matches** | |

### DFL

The adapter maps ball status 0 to dead play. Ball status 1 with possession team 1 or 2 becomes home or away possession. Other combinations are ignored as training targets. These provider fields are targets only: they are not passed to the model as features. Tracking and metadata supply observed coordinates, team membership, and goalkeeper roles.

### SkillCorner

We use the supplied phases of play to identify live intervals and the team in possession. Their complement within official match periods supplies dead play. Unknown owners and conflicting overlaps are excluded. This uses provider phase semantics; it does not interpret every absent coordinate as a stoppage.

A separate concern was coordinate provenance. SkillCorner files can contain inferred positions even when the ball or player was not detected. We explicitly retain only records marked `is_detected=True` with finite x/y. We do not interpolate, extrapolate, or forward-fill coordinates. This keeps missing observations visible to the model instead of training it on a completed view that would not be available in our target game.

### Metrica

Metrica's sample games provide event annotations rather than a directly supplied continuous frame-level possession target. We reconstruct a state from event team, type, subtype, period, and timing. Live actions and restarts establish team possession; ball-out events, relevant shot outcomes, fouls without advantage, and explicit stoppages establish dead play. State is carried between events and reset at each half. Saved shots and ordinary ball losses do not by themselves mark the game dead.

For ball-out and relevant shot events, the event end is used as an approximate dead onset because the event start can precede the boundary crossing. This reconstruction is useful supervision, but event spacing and timestamp semantics limit its precision. We do not describe it as supplied frame-by-frame ground truth. Game 3's changing tracking-column formats are mapped across substitutions. Where goalkeeper roles are unavailable in Games 1 and 2, a documented track-distribution heuristic supplies that feature.

All three sources are converted to the same player and scene features, physical coordinate scales, 5 Hz observation grid, and 25 Hz target/scoring grid. The established external-only ±5s configuration contains 560,076 training windows, with 70,352 validation and 69,412 test centres. Both completed runs record their actual counts and source audits during preparation. Provider annotations, reconstructed event targets, and the Barcelona comparison reference remain distinct sources of evidence.

## 5 Why we changed the ball augmentation

Adding more games does not automatically fix an observation mismatch. In our October development discussions, we first addressed DFL's much more complete ball tracking by hiding the ball during dead play. We then identified the opposite failure: an invisible ball can still be live when held by a goalkeeper. These observations motivated separate augmentation policies for stoppages and live goalkeeper control.

DFL can retain a visible ball throughout dead play, including outside the pitch, whereas the target recording can lose it when the camera moves away. SkillCorner requires a different preparation step: discard inferred positions and keep only detections before adding masking, preserving its natural camera-related gaps. Metrica also retains its natural missing observations, with dead intervals reconstructed from event annotations rather than a direct continuous status field. Additional masking removes available observations; it never fabricates coordinates or fills gaps.

For the revised experiments, we measured the target observation process rather than assuming stoppages were almost entirely invisible. Across 1,275 overlapping ±5s windows wholly inside automatic set-piece tags and outside the public evaluation interval, 74 windows (5.80%) had no ball observations, 708 (55.53%) had observations throughout, and 493 (38.67%) mixed sightings and gaps. Visibility includes tactical and geometry-accepted broadcast positions, including outside the pitch. These automatic tags are observation proxies, not verified dead-play intervals; overlapping windows are correlated, and short tags cannot contribute a full context.

The submitted models add ordinary dead-play masking only to DFL, using 90% hiding of its available ball observations. This leaves approximately 10% visible before whole-window cases and upward integer rounding, close to the 9.4% availability measured against our revised dead-play comparison labels. That reference rate is partly determined by the missing-ball rule, so this is a practical masking target rather than exact ground-truth calibration.

Empirical burst/gap templates guide the hidden regions; they can leave brief boundary-crossing or restart sightings. Approximately 5.8% of wholly-dead DFL windows can have the ball hidden throughout. The measured fully-visible category does not bypass DFL masking, because it would retain too much of DFL's original complete tracking. Live portions of transition windows are protected. SkillCorner and Metrica receive no additional ordinary dead-play masking: their natural dead-ball availability is already approximately 21.0% and 3.7%, respectively, whereas DFL supplies coordinates throughout. Natural gaps are never filled.

The live goalkeeper-control policy is unchanged. Candidate control requires matching live/team targets, a goalkeeper within 2 m of the ball in a penalty area, relative speed at most 2.5 m/s, and at least one second of sustained evidence from the same track. SkillCorner provider possession naming a goalkeeper is also accepted when consistent with live/team targets. In each candidate run, at least 75% of observed ball tokens are hidden contiguously, without changing the player coordinates or target labels. This supplies counterexamples to the shortcut that an invisible ball means dead play.

DFL additionally retains its existing occasional general live masking, selecting 15% of live runs and hiding 20–60% of a selected run. SkillCorner and Metrica do not receive this extra general-live transformation. No source receives new coordinate interpolation, camera crops, player dropout, reflections or team swaps in these experiments.

Every mask removes ball position, velocity, visibility, source and boundary features, as well as player-relative ball features. Missing age is recomputed and velocity validity is reset after reappearance, preventing the model from recovering a hidden ball through another feature. Neither validation nor test receives synthetic masking. Barcelona observation-only visibility patterns inform the augmentation policy; Barcelona possession labels and player-coordinate windows are not supplied as training examples.

Player observations are retained as supplied, including natural gaps. No synthetic player hiding or camera cropping is applied in either submitted model. Ball masking still clears ball-derived player features consistently.

## 6 The submitted model and possession rule

Each visible player is encoded by the same two-layer network with 32-dimensional outputs. Inputs include position, velocity, goalkeeper status, ball-relative position and distance, and observation-validity features. For each team at each time step, masked mean and maximum pooling form an order-invariant team representation. Concatenating the two team representations with ten ball/scene features gives the input to a 64-dimensional temporal Transformer with two layers, four attention heads, and a 128-dimensional feed-forward block. A centre-token classifier predicts home possession, away possession, or dead ball. Team membership and physical position matter; player slot order does not.

Two independently trained checkpoints use this same architecture with ±5s and ±10s context. Both train only on the 24 external games above. AdamW uses learning rate 0.0005, weight decay 0.01, inverse-square-root class weights, and gradient clipping at norm 1. Runs allow 100 epochs, require 20, and stop after 12 epochs without validation improvement. Effective batch size is 128. Each completed 20 epochs. The ±5s checkpoint was selected at epoch 4; ±10s at epoch 8. Selection uses equal mean DFL/SkillCorner validation spellF1, with mean macro-F1 as tie-break. Whole matches are held out.

We infer at 4 Hz and expand centre predictions to 25 Hz by nearest-centre assignment. Input contexts are clipped at half boundaries, with motion dependencies reset at the first sampled frame. Each model is smoothed independently: repeatedly absorb the shortest non-X run shorter than 50 frames into its longer adjacent non-X neighbour, preferring the previous neighbour on a tie. Exactly two-second runs survive. Smoothing never crosses halves or X intervals.

The submitted rule is **dead if either smoothed model predicts dead; otherwise keep the smoothed ±5s label**. There is no further smoothing after the merge. Loose-ball ownership, challenges, and live goalkeeper possession are inferred by the networks from the external supervision and temporal observations. The three-second set-piece and five-second missing-ball rules described above belong to the diagnostic pseudo-label reference; they do not overwrite final predictions. The goalkeeper-control augmentation remains unchanged across both training runs.

## 7 Completed evaluation and choice of submission

The ranking metric, spellF1, matches home/away possession runs one-to-one by temporal IoU and averages F1 over thresholds 0.3–0.7. Dead-ball intervals are not possession spells. We distinguish external supervised evaluation from agreement with our algorithmic Barcelona reference.

| Frozen system | External test spellF1 | Barcelona pseudo agreement, raw | Barcelona pseudo agreement, 2s |
|---|---:|---:|---:|
| Original ±5s | 0.5472 | 0.4382 | 0.5926 |
| Original ±10s | 0.5544 | 0.4119 | 0.5356 |
| ±5s with ±10s dead override | Not measured | 0.4310 | 0.5865 |

External test scores equally weight the DFL and SkillCorner source scores. The merged external test result has not been measured and is not inferred from the individual scores. Barcelona comparisons use the two blocks outside the public five-minute interval, with 143,212 graded reference frames. They are diagnostic agreement scores, not verified possession accuracy. They were generated after checkpoint selection.

We chose the combined system to favour the additional stoppage calls from the longer context while keeping the shorter model's team assignment. This is a design preference, not an empirical claim that the merge wins: its smoothed pseudo-label spellF1 is slightly below ±5s alone (0.5865 versus 0.5926), and its dead-class F1 is also lower (0.8985 versus 0.9114). The public five-minute labels were inspected during earlier pipeline development and are not described as an untouched test. The submission generation command never opens public possession ground truth or comparison pseudo-labels.

The submission runs on both complete match halves, including the public interval. Diagnostic files excluded that interval, so they are not submitted directly. Full-half inference and smoothing can change predictions near the old diagnostic block boundary. The table describes frozen diagnostic results, not a newly scored full-match submission. No scores from the older Barcelona-trained ±10s model are attributed to the submitted ensemble.

## 8 Reproduction and limitations

The accompanying repository provides frozen inference checkpoints, pinned dependencies, observation preprocessing, and the exact command in its README. Place the licensed assets in `data/tactical/` and `data/broadcast/`, install the dependencies, then run:

```sh
python reproduce.py --output outputs/submission.csv
```

This aligns feeds from player geometry, retains geometry-accepted broadcast ball sightings including those outside the pitch, builds observation-only features, runs both frozen models on the complete halves, smooths each separately, applies the dead override, and validates the resulting 199,319-row CSV. Match tracking, videos, feature caches, and reconstructed tracking exports are excluded from the public repository. A manifest records checkpoint and final-file hashes. Training configs, source splits, and completed histories accompany the weights; external dataset URLs and checksums are documented separately.

The two submitted models contain no Barcelona pseudo-label training examples or checkpoint-selection targets. Automatic Barcelona visibility statistics did inform the DFL masking policy, and algorithmic Barcelona pseudo-labels informed diagnostic comparisons. Metrica event-derived supervision remains part of training and has approximate transition timing.

The principal limitations are dataset-specific missingness, imperfect tracking identities and coordinates, uncertain event-derived label timing, potentially incorrect goalkeeper-control candidates, and smoothing that can remove genuine short possessions. A dead override can introduce false stoppages; the diagnostic scores show this tradeoff. The experiments use one seed and a small set of independent held-out matches. Both models use future observations and therefore describe an offline method. Video was used for development inspection, not as a manually annotated source of target labels.

## References

- [Submission requirements](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md).
- [Evaluation protocol](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md).
- [Competition rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md).
- [DFL public release](https://figshare.com/articles/dataset/28196177).
- [SkillCorner open data](https://github.com/SkillCorner/opendata).
- [Metrica sample data](https://github.com/metrica-sports/sample-data).
