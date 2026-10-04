"""Entry point. One capture, one command.

    python run.py run --capture <folder> --tier lidar --out runs/<name>
    python run.py run --capture <folder> --tier video --out runs/<name>
    python run.py run --capture <photo-folder> --tier photo --out runs/<name>
    python run.py repeat --capture <folder> --out runs/<name>_repeat
    python run.py self-check
"""

from cozmo_scan.pipeline import main

if __name__ == "__main__":
    main()
