# Documentation

Start with [current status](status.md) or the [English](../README.md) /
[Chinese](../README-CN.md) quick start. Current guides describe supported usage;
archives preserve dated evidence and are not an active task queue.

| Guide | Contents |
| --- | --- |
| [Setup](guides/setup.md) | Python/CUDA dependencies, Windows connection, Git LFS |
| [Student](guides/student.md) | Model lineage, 85D inputs, training and accepted evidence |
| [Models](guides/models.md) | Accepted models, tracked weights, rejected candidates, legacy commands |
| [Evaluation](guides/evaluation.md) | Entry points, paired protocol and result interpretation |
| [Runtime](guides/runtime.md) | Sync, transport, observation timeout and limits |
| [Training](guides/training.md) | Historical recipes and resume/replay contracts |
| [Results](results/README.md) | Compact versioned summaries of the main evaluations |
| [Experiment index](experiment-log.md) | Topic files and legacy section navigation |
| [Archive](archive/README.md) | Historical experiments and session notes |

The pre-reorganization snapshot is Git commit `2f150fb`. Full local camera/
telemetry caches remain outside Git; selected model weights and compact results
are versioned separately.

Keep usage and decisions in the guides and status page. Add dated measurements
to the relevant archived topic, link them from the experiment index, and split
topics before they grow into another monolithic log. Update both root READMEs
when the accepted model or quick-start command changes.
