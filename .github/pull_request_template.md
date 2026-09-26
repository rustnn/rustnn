## Summary

- [ ] Describe the user-visible and code-level changes.

## Validation

- [ ] `make test`
- [ ] Relevant WPT or integration checks

## Documentation

See `docs/development/documentation-policy.md` for which page belongs to which change.

- [ ] Updated the rustdoc comments and docs pages that describe the changed behavior (or no documented behavior changed)
- [ ] If backend converter/executor operator support changed, ran `make docs-backend-ops` and committed `docs/development/backend-operator-support.md`
- [ ] `make docs-api` and `make ci-docs` pass if rustdoc or docs pages changed

