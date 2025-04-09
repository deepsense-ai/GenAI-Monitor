### Conventions
1. Use the commit message format consistent with the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/#summary) standard
2. Use typing but omit mypy for static type checking due to the "hacky" nature of some of the things in hidden mode
3. Use typing structures for python <3.10 to avoid compatibility issues. Use:
   ```
   from typing import List
   def foo(ints: List[int]):
       pass
   ```
   Instead of:
   ```
   def foo(ints: list[int]):
       pass
   ```
4. Use Google-type docstrings, omit type annotations for parameters if provided in function/method signature
5. Use pre-commit configured in this repo, feel free to propose changes to the configuration if it feels too restrictive
6. Add tests to PRs when possible
7. When in doubt about anything python, consult [this](https://google.github.io/styleguide/pyguide.html) style guide, there's probably a reasonable approach there
8. Rebase your feature onto main before merging, if there are many atomic commits consider squashing upon merge