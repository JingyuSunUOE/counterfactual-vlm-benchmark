# Test Utilities

This directory contains reusable smoke tests and validation scripts for the
project. The default tests are designed to be safe: they use `--help`,
`--dry-run`, static path checks, or report aggregation, and they should not call
model APIs or write benchmark outputs.

## Safe Smoke Tests

```bash
python test/fashion_industry/test_eval_cli.py
python test/fashion_industry/test_gen_cli.py
python test/fashion_industry/test_report_paths.py
python test/if_exist/test_eval_cli.py
```

## Live API Test

`test/api/test_closed_api_keys.py` sends one small live request per selected
provider. Run it only when API connectivity should be tested:

```bash
python test/api/test_closed_api_keys.py --provider all
```

The live API test loads the repository `.env` with `override=True`.
