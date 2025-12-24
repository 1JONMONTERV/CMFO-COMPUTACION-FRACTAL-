# CMFO System Reproductions

This directory contains documentation of CMFO reproductions across different execution environments.

## Purpose

These reports demonstrate that CMFO can be installed and executed in clean environments without author intervention, confirming:

- Deterministic behavior
- Reproducible results
- Platform independence
- No hidden dependencies
- Automated execution compatibility

## Available Reproductions

### Automated Systems
- [Antigravity](antigravity.md) - Google Cloud automated execution (2025-12-16)

### Manual Reproductions
- Human reproductions documented in main repository

## Verification Procedure

To independently verify any reproduction:

1. Clone repository at specified commit
2. Install dependencies as documented
3. Execute commands as listed
4. Compare results with reported output

## Reproduction Standards

Each report includes:
- System specifications
- Exact commit hash
- Installation procedure
- Commands executed
- Test results
- Execution metrics
- Verification method

## Adding New Reproductions

To document a new reproduction:
1. Use template format (see existing reports)
2. Include all required sections
3. Provide verifiable commit hash
4. Document exact environment
5. Include pass/fail results

## Status

Current reproduction coverage:
- Automated systems: 1
- Operating systems: Windows, Linux (via containers)
- Python versions: 3.13.0
- Test coverage: 95% pass rate

**Last Updated**: 2025-12-16
