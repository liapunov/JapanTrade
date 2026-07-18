# Changelog

## Unreleased

- Enforced complete current and prior comparison windows with country/month diagnostics.
- Added normalized primary-key conflict detection and bounded MCP table responses.
- Consolidated development dependencies around `pyproject.toml` and a committed `uv.lock`.
- Removed import-time root logging configuration and its automatic log file.
- Added a dedicated Codex MCP plugin guide and refreshed the root workflow and troubleshooting documentation.
- Modernized the tutorial notebooks around the current package, CLI, direction-safe data model, and analyst APIs.

## 0.1.0 - 2026-07-11

- Added a public `japantrade` CLI for downloading, preparing, and querying HS trade data.
- Added direction-aware normalized records so imports and exports cannot overwrite each other.
- Added HS keyword search, country category rankings, and multi-country product comparisons.
- Added Parquet preparation metadata and packaged country/HS lookup data.
- Established the local-dataset, HS/value-only v1 scope; PC classification, quantity comparisons, hosted data, and the dashboard remain future work.
