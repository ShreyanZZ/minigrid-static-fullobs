import os

# CONFIGURATION
USER = "shreyan"        # Your IITK username
HOST = "subhajit1.cse.iitk.ac.in"  # The server address
REMOTE_PATH = "~/project/" # Where to save it

# COMMAND (Using SCP because it is built into Windows 10/11)
# Note: We use scp here because rsync is hard to get on Windows without Git Bash.
command = f"scp -r . {USER}@{HOST}:{REMOTE_PATH}"

print("--> Uploading files...")
os.system(command)
print("--> Done!")