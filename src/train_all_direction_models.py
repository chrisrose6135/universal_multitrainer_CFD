from __future__ import annotations

# Convenience entry point. The main trainer already trains all configured symbols
# when --symbols is omitted.
from .train_direction_policy import main


if __name__ == '__main__':
    main()
