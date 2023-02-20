import subprocess
# Similarly to os.system()
# subprocess can execute a shell command, but:
# - we need to pass each command token in a list of strings
# - we get back the output!
# The advantage is we can persist the output
# as a witness of the current state

# pip check
broken = subprocess.check_output(['pip', 'check'])
with open('broken.txt', 'wb') as f:
    f.write(broken)

# pip list --outdated
outdated = subprocess.check_output(['pip', 'list','--outdated'])
with open('outdated.txt', 'wb') as f:
    f.write(outdated)

# python -m pip show numpy
numpy_info = subprocess.check_output(['python','-m','pip', 'show', 'numpy'])
with open('numpy.txt', 'wb') as f:
    f.write(numpy_info)