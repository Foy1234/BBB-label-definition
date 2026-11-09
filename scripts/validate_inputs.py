# validate_inputs.py — Validates the frozen parent tables and Stage II targets through the bundled read-only research reproduction checks.
from verify_release import main
import json
print(json.dumps(main(),indent=2))
