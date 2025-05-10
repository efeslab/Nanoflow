## How to build docs

1. Install `doxygen`.

    ```bash
    $ sudo apt-get install doxygen
    ```

2. Install Python packages below. If you install them on the user's local, you need to include `~/.local/bin` to `$PATH` (to use `sphinx-build`).

    ```bash
    $ sudo python3 -m pip install sphinx sphinx_rtd_theme breathe
    ```

3. Create Doxygen documents.

    ```bash
    $ doxygen
    ```

4. Create Sphinx documents.

    ```bash
    $ sphinx-build -b html -Dbreathe_projects.mscclpp=$PWD/doxygen/xml $PWD $PWD/sphinx
    ```

5. Done. The HTML files will be on `sphinx/` directory.
