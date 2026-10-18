# Third party notices

Model tensors are extracted from NVIDIA's pinned Nemotron Super checkpoint under
OpenMDW-1.1. `LICENSE` is the unmodified official agreement retrieved from
https://github.com/OpenMDW/OpenMDW/blob/b26b32b34ad2edcc29a7707abb68dcfb25a538c1/1.1/LICENSE.OpenMDW-1.1.
The pinned parent model card links to this license. Its repository has no
LICENSE file. The old Nano Omni NVIDIA license was not copied.

The loader uses remote C-RADIO implementation code at
`nvidia/C-RADIOv4-H@0057b339059c0b9e1b4ba996f975410ebbfdfcc8`, and code/configuration
from `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16@e5e9932441de940c9a62185c870ea5bcd4cd24e2`.
Those remote files retain their own NVIDIA copyright and license notices and
are fetched separately by Transformers. They are not relicensed here.
No weights from either code dependency are included.
