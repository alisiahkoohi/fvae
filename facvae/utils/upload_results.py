import subprocess

from facvae.utils import checkpointsdir, gitdir, logsdir, plotsdir


def upload_results(args, flag=""):
    repo_name = "factorialVAE"
    bash_commands = [
        "rclone copy " + flag + " " +
        checkpointsdir(args.experiment, mkdir=False) + " MyDropbox:" +
        repo_name + '_nature' +
        checkpointsdir(args.experiment, mkdir=False).replace(gitdir(), ""),
        "rclone copy " + flag + " " +
        logsdir(args.experiment, mkdir=False) + " MyDropbox:" +
        repo_name + '_nature' +
        logsdir(args.experiment, mkdir=False).replace(gitdir(), ""),
        "rclone copy " + flag + " " +
        plotsdir(args.experiment, mkdir=False) + " MyDropbox:" +
        repo_name + '_nature' +
        plotsdir(args.experiment, mkdir=False).replace(gitdir(), ""),
    ]

    for commands in bash_commands:
        process = subprocess.Popen(commands.split())
        process.wait()
