 #!/bin/bash
 #
 # hello-world.sh
 # My CHTC job
 #
 # print a 'hello' message to the job's terminal output:
 echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"
 #
 # keep this job running for a few minutes so you'll see it in the queue:
 sleep 180