# Portable evaluation summaries

These compact JSON files are versioned so the principal results can be reviewed
without local camera/telemetry archives. Each records the source run and original
results-file SHA256, checkpoint hashes, saved menu settings, input contract,
episode metrics and audit outcome. Host addresses, raw state and images are omitted.

| File | Experiment |
| --- | --- |
| [student-24x2000.json](student-24x2000.json) | Original camera-only student, 24 new appearance layouts, 2k cap |
| [student-10x3000.json](student-10x3000.json) | Same original student, 10 further layouts, 3k cap |
| [student-paired-4.json](student-paired-4.json) | Four shared layouts comparing original and bend candidate |

Different caps and unpaired batches stay separate. Saved menu settings do not
independently prove live engine state. Original full logs, model-input arrays,
source snapshots and camera-replay scripts remain local-only under the recorded
`logs/` run paths; these JSON files do not replace the raw audit evidence.
The [status](../status.md) and [experiment index](../experiment-log.md) explain decisions.
