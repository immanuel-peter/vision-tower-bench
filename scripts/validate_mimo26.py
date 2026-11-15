"""Compare a MiMo 2.6 release with the pinned parent's own vision class.

The reference imports MiMoVisionTransformer from the SHA-pinned parent remote
code and never constructs the parent language model. That class cannot strictly
load its own checkpoint: it creates three merger biases the checkpoint lacks.
The reference sets only those biases to zero and leaves the forward untouched.
"""

from vtb.validate_mimo26 import main

if __name__ == "__main__":
    main()
