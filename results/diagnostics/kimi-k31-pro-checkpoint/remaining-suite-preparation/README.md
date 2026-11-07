# Remaining suite preparation

These scripts are prepared for a future explicit GPU reservation. They have
not produced measurements. The inventory audit checks 98 pending primary cells,
27 dense comparisons and eight later CPU correspondence comparisons.

The owning worktree still runs CPU8 raw attention sequentially. Finish that
command normally before CUDA. A new gpu-grant.json must explicitly record the
physical GPU, active=true and phase remaining-suite or remaining-dense or
remaining-correspondence-throughput. Old extraction and parity grants do not
permit these jobs. Invoke gpu_remaining_suite.py through run_logged.py --gpu
so the full process holds the physical device flock. All model execution uses
logical cuda:0, eight threads, at most two loaders and batch one for vision.

The runner validates complete output contents before reuse, records actual
hardware and source hashes, and updates only verified inventory entries.
It retains full DIODE/KITTI grids and the separate CPU geometry cache provenance.
It reruns actual selected correspondence IDs before inference and validates
all native slices for each image. Raw attention uses its separate active CPU
runner and is excluded from this GPU queue. Preserve identical actual device
and thread settings within unfinished sibling controls when transitioning.

After the granted scope finishes, run release_remaining_gpu.py under CVDempty
to reacquire the lock, verify zero memory and no CUDA processes, revoke the
recorded grant and report release. Run the eight CPU8 correspondence pairs
with CVDempty after source measurements exist and no other CPU training job
is active. Source-backed SPair zero-keypoint rows remain in all 3,600 selections;
paired audits report the actual finite intersection and exclusion union.

Production official TORCH_SDPA BF16 parity remains failed. The independent
mathematical-reference conversion acceptance and original tolerances remain
unchanged. None of this preparation completes a benchmark inventory cell.
