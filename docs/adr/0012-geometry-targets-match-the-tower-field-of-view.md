# Geometry targets are cropped to the square the Tower saw

`vtb.images.square_crop` resizes an image's short side to 448 and centre-crops, so a
768 by 1024 DIODE frame reaches every Stage as its middle 768 by 768. The geometry runner
was scoring predictions against the full 768 by 1024 target, upsampled to fit. A quarter
of every scored pixel therefore sat outside the field of view the Tower was given, and no
Stage could do better than guess there.

That does not favour one Stage over another, so the headline comparison would have
survived it. It does add a fixed noise floor to every cell, which compresses the gap the
run exists to measure, and it makes the absolute numbers uninterpretable against any
published depth result. The runner now crops each target to its centre square before the
metric sees it. The crop is exact rather than resampled: at a 448 short side the 448 pixel
crop maps back to 768 original pixels, so the ground truth keeps native resolution and
loses nothing to interpolation.

The prep step still writes full-frame targets. The crop belongs to the runner, which is
where the input geometry is known, and keeping the npz uncropped means a change to
preprocessing does not require re-preparing 6.3 GB.

Two related defects in the same runner, both found before any cell ran. The cache holds
bfloat16 and the heads are float32, so the unmatched arm raised on its first convolution;
only the capacity-matched arm worked, because PCA returned float32 by accident. And the
targets were re-read from the npz once per cell, which is ten reads of 6.3 GB for a
MoonViT-V2 run whose slices all hold the same images in the same order.
