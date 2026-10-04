# polcheck

Regression testing for robots that run learned policies. Compares a new policy version's simulated runs against the previous version's and answers: is it meaningfully worse, where, and how sure are we?

Early development. The build spec is [.agents/BUILD_PLAN.md](.agents/BUILD_PLAN.md); design decisions are in [docs/DECISIONS.md](docs/DECISIONS.md).

## Development

```sh
uv sync
uv run pytest
uv run ruff check
uv run mypy src
uv run pre-commit install   # optional: run the checks on every commit
```
