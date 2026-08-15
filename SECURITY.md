# Security Policy

## Reporting a Vulnerability

If you find a security vulnerability — including accidentally committed
secrets, credentials, or license material — **do not open a public issue**.

Please report it privately by emailing the repository maintainers (the address
is listed on the repository page under *About → Manage → security*, or contact
the owner via GitHub). Include:

- a description of the issue
- the affected files/commits, if known
- any suggested fix

We will acknowledge reports within 7 days and work on a fix before public
disclosure where possible.

## Secret Hygiene

- Never commit `.env`, `license.dat`, API keys, or credentials.
- If a secret was ever committed, assume it is compromised: rotate it, then
  report it so the history can be scrubbed.
- CI workflows must not require repository secrets to pass; secrets are only
  needed for manual release steps.

## Supported Versions

Only the `master` branch is actively maintained. Released versions are tagged
and receive fixes on a best-effort basis.
