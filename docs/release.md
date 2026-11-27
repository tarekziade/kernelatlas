# Release KernelAtlas to PyPI

KernelAtlas uses the same branch and tag convention as mlinter. A push to a branch such as `v0.1-release` builds and tests the distributions. Pushing the matching version tag publishes those artifacts through GitHub Actions.

## First release setup

1. In the GitHub repository `tarekziade/kernelatlas`, create an environment named `pypi-release`. Configure release approvals there if desired.
2. In your PyPI account's **Publishing** settings, add a **pending GitHub publisher** with these values. For an existing PyPI project, use its **Publishing** settings instead.

   | PyPI setting | Value |
   | --- | --- |
   | Project name | `kernelatlas` |
   | Owner | `tarekziade` |
   | Repository | `kernelatlas` |
   | Workflow filename | `release.yml` |
   | Environment | `pypi-release` |

The first successful upload creates the PyPI project. Authentication uses GitHub OIDC. [PyPI first-project setup](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)

## Verify a release

Set the version in `pyproject.toml` and update `CHANGELOG.md`. From the repository root, in a Python 3.11+ virtual environment, run:

```sh
python -m pip install -e '.[profile,release]'
make test
make build-release
make check-release
```

The build produces a source distribution and a wheel built from that distribution. The check validates metadata with Twine and installs the wheel in a fresh environment outside the checkout. It exercises both CLI entry points and checks the packaged Metal shader. Set `RELEASE_TAG=v0.1.0` when invoking `make check-release` to verify the tag locally.

## Publish

Push the reviewed changes to GitHub. For the first release, create the release branch from the commit to ship and wait for its Release workflow to pass:

```sh
git switch -c v0.1-release
git push -u origin v0.1-release
```

From that verified commit, push the version tag to publish:

```sh
git tag -a v0.1.0 -m 'Release 0.1.0'
git push origin v0.1.0
```

For later patches, fast-forward the existing release branch and use the new version in the tag. The tag build checks that the tag matches the distribution version, then the publish job downloads exactly the artifacts from that build. After it succeeds, verify the release in a fresh environment with `python -m pip install 'kernelatlas==0.1.0'` and `kernelatlas --version`. Update the changelog and version on main for the next release.
