import subprocess
# Similarly to os.system()
# subprocess can execute a shell command, but:
# - we need to pass each command token in a list of strings
# - we get back the output!
# The advantage is we can persist the output
# as a witness of the current state

# pip list
# python -m pip list 
broken = subprocess.check_output(['python', '-m', 'pip', 'list'])
with open('installed.txt', 'wb') as f:
    f.write(broken)

# pip freeze
# python -m pip freeze
# python -m pip list --format=freeze
#outdated = subprocess.check_output(['python', '-m', 'pip', 'freeze'])
outdated = subprocess.check_output(['python', '-m', 'pip', 'list', '--format=freeze'])
with open('requirements.txt', 'wb') as f:
    f.write(outdated)

# python -m pip show scikit-learn
sklearn_info = subprocess.check_output(['python', '-m', 'pip', 'show', 'scikit-learn'])
with open('sklearn_info.txt', 'wb') as f:
    f.write(sklearn_info)