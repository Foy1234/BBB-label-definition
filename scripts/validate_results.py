# validate_results.py — Rechecks the published training metrics and external PAMPA prediction statistics using the original seed-by-molecule result tables and fixed study evaluation identities.
from verify_release import main
import json
print(json.dumps(main(),indent=2))
