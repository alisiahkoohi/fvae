import subprocess
import os
from .project_path import gitdir, checkpointsdir, plotsdir


def upload_to_dropbox(args, flag: str = '--progress --transfers 8'):
    """
    Uploads experiment data and plots to Dropbox using rclone.

    Args:
        args (Namespace): Namespace object containing experiment information.
        flag (str, optional): Additional rclone flags for the upload command.
            Defaults to '--progress --transfers 8'.
    """
    # Get the base name of the Git repository
    repo_name = os.path.basename(gitdir())

    # Define cloud paths for data and plots in Dropbox
    cloud_data_path = os.path.join(repo_name, 'data', 'checkpoints',
                                   args.experiment)
    cloud_plots_path = os.path.join(repo_name, 'plots', args.experiment)

    # Define rclone upload commands for data and plots
    bash_commands = [
        'rclone copy ' + flag + ' ' +
        checkpointsdir(args.experiment, mkdir=False) + ' MyDropbox:' +
        cloud_data_path,
        'rclone copy ' + flag + ' ' + plotsdir(args.experiment, mkdir=False) +
        ' MyDropbox:' + cloud_plots_path
    ]

    try:
        # Execute rclone upload commands
        for command in bash_commands:
            process = subprocess.Popen(command.split())
            process.wait()
    except:
        print("Could not upload experiment data to Dropbox!")
