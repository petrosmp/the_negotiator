This directory contains the logs emitted by TheNegotiator agent during the TUC internal
competition.

The files in this directory are the following:
    - TheNegotiator.log
        The original log file. Since logging is done by appending to the log file, and a
        lot of errors happened while trying to run the agent in the tournament, this file
        has more thn 883.000 lines, rendering it unusable. This is why the rest of the
        files exist.

    - TheNegotiator_no_tracebacks.log
        The original file, keeping only "meaningful" log entries emitted by TheNegotiator
        through logging.log(), i.e. no tracebacks. It was obtain by running the following
        command:
            cat TheNegotiator.log | grep " - TheNegotiator - " > TheNegotiator_no_tracebacks.log

    - cleanup_logs.py
        The no_tracebacks file is still huge (~196.000 lines) because (for some reason) a lot
        of lines are repeated. For this reason we created this script to remove duplicate lines.

    - TheNegotiator_no_repetition.log
        This file is finally pure logs, around 3.000 lines of them. Enjoy!
