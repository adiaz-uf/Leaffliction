# Common Requirements to Pass `flake8`

`flake8` checks your code in three main areas. To "pass" the linter, you must generally satisfy all of these points:

---

## 1. Code Style (pycodestyle / PEP 8)

This is the most common reason `flake8` fails. It checks that your code follows the official Python style guide (PEP 8).

* **Indentation:** Use 4 spaces for indentation. (`E111`)
* **Line Length:** Keep all lines under 79 characters. (`E501`)
* **Whitespace:**
    * Use spaces around operators (e.g., `x = y + 1`, not `x=y+1`). (`E225`)
    * Put a space after commas (e.g., `[1, 2, 3]`, not `[1,2,3]`). (`E231`)
    * Avoid trailing whitespace (spaces at the very end of a line). (`W291`)
* **Blank Lines:**
    * Use two blank lines to separate top-level functions and classes. (`E302`)
    * Use one blank line to separate methods inside a class. (`E301`)
* **Imports:**
    * All imports must be at the top of the file. (`E402`)
    * Imports should be grouped (standard library, then third-party, then local).
    * Do not import multiple modules on one line (e.g., `import os, sys`). (`E401`)
* **Naming Conventions:**
    * Use `snake_case` (all lowercase with underscores) for functions, methods, and variables.
    * Use `PascalCase` (uppercase first letter of each word) for classes.
* **Comparisons:**
    * Use `is not` instead of `not ... is`. (`E713`)
    * Use `is None` or `is not None` when checking for `None`. (`E711`)

---

## 2. Logical Errors (PyFlakes)

This section checks for "bugs" or code that is syntactically correct but likely a mistake.

* **Undefined Variables:** You must define a variable before you use it. (`F821`)
* **Unused Imports:** If you `import a_module`, you must use `a_module` somewhere in that file. (`F401`)
* **Unused Variables:** If you define a variable (e.g., `my_var = 10`), you must use it later. (`F841`)
* **Duplicate Keys:** You cannot define the same key twice in a dictionary (e.g., `{'a': 1, 'a': 2}`).

---

## 3. Code Complexity (McCabe)

This checks if your functions are too complex, which makes them hard to read and test.

* **Cyclomatic Complexity:** Your functions and methods must not be "too complex." (`C901`)
    * This usually means you have too many `if`, `elif`, `else`, `for`, `while`, `and`, or `or` statements nested inside a single function.
    * **To fix this:** Break the large function down into smaller, simpler functions.