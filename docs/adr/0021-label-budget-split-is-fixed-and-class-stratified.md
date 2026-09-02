# Label budgets change only the semantic training split

Hypothesis 3 names label efficiency alongside token count and latency. A label-budget run
must therefore change only the number of labelled training examples. Validation and test
images must remain identical or the cross-budget ranking is not comparable.

`vtb.probe_run --label-fraction` starts from the existing fixed-seed 70/15/15 split and
subsamples only `split.train`, at seed 0, in proportion to class frequency. It keeps an
exact rounded global count when that count can cover every represented class. When it
cannot, it keeps one example per class. On the 13,000-image ImageNet-100 validation export,
the full training split contains 9,100 images. A literal one-percent count would be 91 and
could not cover 100 classes, so the one-percent condition retains 100 images, an effective
1.10 percent. The output records requested fraction, actual training count, and full
training count.

The capacity-matching reducer is fit only on the budgeted training indices, as is the
readout. This follows the existing meaning of `split.train` and prevents unlabelled examples
from silently entering one part of the training path.

`scripts/label_budget_matrix.sh` restricts each cache to its deepest `tower` Stage, the
matched arm, and both attention and mean readouts. It runs uniform readout phases over
fractions 0.01, 0.05, 0.20, and 1.0. Six Towers therefore produce exactly 48 one-cell
outputs. Tests verify deterministic selection, complete class coverage at the smallest
budget, proportional allocation above that minimum, and unchanged validation and test
indices.

## Result

No label-budget experiment ran. The correspondence stop condition fired while the GPUs
were completing workstream A, before semantic cache extraction, the 48 probe cells, or the
required idle-box latency pass could start. There are no Tower rankings at one percent or
100 percent, no measured token-count and latency table, and no evidence for hypothesis 3
from this run.

If v1 ships without resuming after the correspondence decision, hypothesis 3 must be
dropped. The implementation is committed only to make a resumed run reproducible; it is
not a result.
