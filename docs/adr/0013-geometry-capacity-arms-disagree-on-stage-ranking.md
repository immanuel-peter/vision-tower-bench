# The geometry capacity arms disagree on Stage ranking

ADR-0008 accepts capacity matching only when model rankings and Relative Depth curves
agree between the matched and unmatched runs, and requires any disagreement to be
reported rather than smoothed over. The first geometry run produces one.

On MoonViT-V2 depth `d1` at the deepest point, averaged over three seeds, the unmatched
arm reads `merged` 0.5246, `projected` 0.4860 and `tower` 0.4575. The matched arm reads
`projected` 0.5579, `merged` 0.5507 and `tower` 0.4971. `merged` and `projected` swap
places. The unmatched gap of -0.0386 is 4.6 seed standard deviations and is not noise;
the matched gap of +0.0071 is one, and is.

The disagreement is 0.0457 wide. On the semantic side the same two arms agreed to within
0.0283 and changed no conclusion, which is why that check passed there and fails here.
The swap is also the whole question: it separates a Projector that appears to destroy
spatial information from one that does not.

Three things point at head capacity rather than at the Stage. The unmatched `projected`
head trains 4,852,480 parameters against 3,279,616 at `merged`, on 541 images. Its
normals counterpart swings 4.4931 degrees across three seeds against 0.1547 at the narrow
`tower` Stage, so the widest cells are also the least stable. And reducing every cell to
512 dimensions removes the instability and the gap together.

That is an argument, not a measurement, so the run does not declare a winner. Both arms
are published. The matched arm is the one ADR-0010 already nominates for the headline, and
the headline claim survives either way: neither arm shows `projected` below `tower` on
either task. What neither arm can settle at 541 training images is the smaller question of
whether the Projector costs anything at all relative to `merged`. Answering that needs the
DIODE training split, which is the decision ADR-0011 parked pending this run.

The `tower` against `merged` control bounds how much either arm can be trusted. That pair
is a lossless regrouping, verified against the cache on disk, and the matched arm still
scores it 0.0536 apart on depth. Any Stage effect smaller than that is below the noise
floor of this readout, and the disputed `merged` against `projected` gap is smaller than
it in both arms.
