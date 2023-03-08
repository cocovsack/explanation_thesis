# Set up
## Python version
Use python version 3.9.7. You can use [pyenv](https://github.com/pyenv/pyenv) to manage multiple python versions.


With pyenv installed, you can install a specific version of python (e.g., 3.9.7) with:
   ```bash
   $ pyenv install -v 3.9.7
   ```

You use following command to set the correct python version for a specific terminal window:
   ```bash
   $ pyenv shell 3.9.7
   ```

## Virtual environment
1. Create a virtual environment with:
   ```bash
   $ python -m venv exp_venv
   ```

2. Activate virtual environment:
   ```bash
   $ source exp_venv/bin/activate
   ```

3. Install requirements with:
   ```bash
   $ pip install -r requirements.txt
   ```

## Trouble shooting

# Run program
0. If you need to set the correct python version, use:
   ```bash
   $ pyenv shell 3.9.7
   ```

1. Activate virtual environment:
   ```bash
   $ source exp_venv/bin/activate
   ```

2. From top folder, run program with:
   ```bash
   $ python src/chefbot_utils/explainable.py
   ```
   
