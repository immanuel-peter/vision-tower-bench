"""Independent pinned-parent BF16 parity. Never constructs a parent LLM.

The reference uses the parent checkpoint's RadioModel and VisionProjector code,
not build_parts or the local release implementation. Explicit key conversion
reproduces the conversion rules shipped in the pinned parent modeling_radio.py.
"""

from vtb.validate_nemotron35 import main

if __name__ == "__main__":
    main()
