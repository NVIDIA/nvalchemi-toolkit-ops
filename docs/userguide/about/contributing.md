<!-- markdownlint-disable MD025 -->

(contributing)=

# Contribution Guide

There are many (welcome) ways to contribute to ALCHEMI Toolkit-Ops:

- All bug reports are welcome; to help maintainers, please fill out the prescribed
the Github issues template with as much information as you can provide, and when
possible, a minimum working example that reproduces the issue.
- Feature requests; if you think of something that you would like as part of the
public API, it might benefit others as well. Submit a feature request via Github
issues.
- Provide general feedback and post discussions in Github Issues; the Github
Discussions may not be effectively monitored, so please post issues instead.
- Contributing bug fixes and new features: our preferred way is to start a
conversation on Github issues to hash out implementation details and to
gauge interest.

## Code Contributions

Code contributions from the community are welcome; rather than requiring a formal
Contributor License Agreement (CLA), we use the Developer Certificate of Origin (DCO) to
ensure contributors have the right to submit their contributions to this project.
Please ensure that all commits have a sign-off added with an email address that
matches the commit author to agree to the DCO terms for each particular contribution.

- Full text of the DCO:

  ```text
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license
    document, but changing it is not allowed.
  ```

  ```text
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to
    submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge,
    is covered under an appropriate open source license and I have the right under that
    license to submit that work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am permitted to submit under a
    different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified
    (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and
    that a record of the contribution (including all personal information I submit with
    it, including my sign-off) is maintained indefinitely and may be redistributed
    consistent with this project or the open source license(s) involved.

  ```

## Developer Set Up

The guide below provides step-by-step instructions to start development locally, up
to submitting a pull request with your bug fixes/features:

1. Create a fork of ALCHEMI Toolkit-Ops through the Github page, or
[here](https://www.github.com/NVIDIA/nvalchemi-toolkit-ops/fork).
2. Clone your fork somewhere in your local system, i.e. via
`git clone git@github.com:<username>/nvalchemi-toolkit-ops.git`
3. Track changes in a new branch: if there is an issue attached to
the work, prepend the issue number to your branch name, which should
be something descriptive (e.g. `15-what-is-fixed`). Complementary to
this would be to create a Git worktree to you to work on multiple
branches concurrently.
4. Make your bug fixes/feature implementation; try and adhere to
general best practices such as:
    - Keep changes isolated, and tightly scoped for your fix. Try
    not to "omnibus" too many peripheral elements to your branch.
    - Make small commits, frequently.
    - We encourage the use of
    [semantic commit messages](https://www.conventionalcommits.org/en/v1.0.0/#summary)
    - Refer to and adhere to the [style guide](#style-guide). This
    provides a fairly extensive set of guidelines to help us ensure
    code quality and consistency across the codebase.
5. When you are finished with your changes, push to your fork
(i.e. `git push -u <remote> <branch-name>`). As a simple checklist,
your changes should:
    - [ ] Pass `pre-commit` hooks, which include style and static
    analysis checks.
    - [ ] Unit tests have been added, and/or updated.
    - [ ] API documentation has been added, and/or updated. Ensure
    sufficient changes maintain sufficient docstring coverage.
6. On your Github fork, submit a pull request. If you are unsure about
any implementation details, mark your pull request as a draft and ask
for maintainer feedback before going through a more 'formal' round of
reviews.
7. Work with maintainers, who will review and provide feedback on details
regarding your changes. The quality and timeliness of the feedback depends
heavily on the complexity of your changes, so please try and keep changes minimal.
    - We will also use code review agents, such as Greptile and Code Rabbit,
    to provide initial feedback. These agents will generally be configured to
    provide very general guidance, such as typographical errors, inconsistencies,
    and so on. **Human review and approval will always be required**, and we
    treat reviews by agents as suggestions, not requirements.
8. If a maintainer provides feedback on your implementation, please address
them accordingly. If and when the changes are satisfactory, reviewers will
approve the pull request and perform a squash-merge.

### Workflow

To make the above more concrete, here are commands you can run for most of the
steps involved:

```bash
# Step 1: Fork the repository on GitHub
# Visit: https://github.com/NVIDIA/nvalchemi-toolkit-ops/fork
# (This is done through the GitHub web interface)

# Step 2: Clone your fork
git clone git@github.com:<your-username>/nvalchemi-toolkit-ops.git
cd nvalchemi-toolkit-ops

# Optional: add upstream to keep local branch synced
git remote add upstream git@github.com:NVIDIA/nvalchemi-toolkit-ops.git

# Step 2.5: Set up development environment; install `uv` if not available already
uv sync --all-extras
uv pip install -r './test/test-requirements.txt'
pre-commit install

# Step 3: create a branch for changes
git checkout -b 15-fix-description

# Step 4: Run `pytest`; `Makefile` in root folder contains definition
# for some of these commands. Run `coverage` tool afterwards.
make pytest
make coverage

# When things pass, add and commit files; make sure to address
# any outstanding pre-commit issues
git add <files>
git commit -s -m "fix: added missing file"

# When done with changes, push
git push -u origin 15-fix-description
# Optionally, if main has been updated remotely
# git fetch upstream
# git rebase upstream/main

# After PR is merged, clean up by updating your local
# main branch and deleting your temporary branch
git checkout main && git pull upstream main
git branch -D 15-fix-description
```

## Building Documentation

ALCHEMI Toolkit-Ops uses `sphinx` to build and serve documentation. The recommended
workflow is to make your changes that touch documentation (e.g. docstrings, markdown
files, examples) and preview them locally in your browser. Assuming you have carried
out the instructions in the prior section, you can run:

```bash
cd docs
make html
# preview the docs by opening `docs/_build/html/index.html`
```
