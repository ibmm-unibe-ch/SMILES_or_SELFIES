import yaml
import subprocess

# Load the fairseq_git.yml file
with open('SMILES_OR_SELFIES.yml', 'r') as file:
    env = yaml.safe_load(file)

# Extract the pip packages
pip_packages = [pkg for dep in env['dependencies'] if isinstance(dep, dict) for pkg in dep.get('pip', [])]

# Uninstall the pip package (just in case)
subprocess.run(['pip', 'uninstall', '-y'] + pip_packages)

# Reinstall the pip packages
subprocess.run(['pip', 'install'] + pip_packages)
