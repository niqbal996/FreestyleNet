srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=10 -p H100 --mem=50000 \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/FreestyleNet:/home/iqbal/FreestyleNet \
  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.09-py3.sqsh  \
  --container-save=/netscratch/naeem/freestyle-torch1.14.sqsh \
  --time=00-02:00 \
  --pty /bin/bash
