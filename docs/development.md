# General Developer Guide

## Formatting and Linting

### Tools

Most of the formatting and linting tools listed below (except for mypy, the Python
static type checker) have been registered as [pre-commit](https://pre-commit.com/)
hooks. The hooks can be installed using the command:

#### Register the pre-commit hooks to run when running `git commit`

```bash
pre-commit install
```

After the installation, those hooks are automatically triggered every time a commit is
made, and are applied on files that are changed in the commit. Use the following command
if you want to run hooks on all files (regardless whether the files are changed or not).

```bash
# Run pre-commit hooks over all the files.
pre-commit run --all-files
```

### Coding Style

Please follow the following coding style guide:

- [Python Coding Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Python Docstring Style Guide](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- [File Name and Type Convention](https://developers.google.com/style/filenames)

### Integration of Checkers with Editors and IDEs

We recommend to integrate frequently used checkers (which were installed with the
pre-commit hooks, as discussed above) into your editors and IDEs, so that they are
automatically run on save (and you can stop worrying about running them manually
yourself). We provide some examples of how they can be set up:

<!-- markdownlint-disable line-length -->

| IDE        | [black](https://github.com/psf/black)                                                                      | [ruff](https://beta.ruff.rs/docs/)                                                           | [mypy](https://mypy-lang.org/)                                                                                 |
| :--------- | :--------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| **VSCode** | [ms-python.black-formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) | [charliermarsh.ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) | [ms-python.mypy-type-checker](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker) |

<!-- markdownlint-enable line-length -->
