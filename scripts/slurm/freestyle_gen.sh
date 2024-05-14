srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=10 -p A100-40GB --mem=50000 \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/FreestyleNet:/home/iqbal/FreestyleNet \
  --container-image=/netscratch/naeem/freestyle-torch1.10.sqsh \
  --mail-type=end --mail-user=naeem.iqbal@dfki.de --job-name=FreestyleNetFlis \
  --container-workdir=/home/iqbal/FreestyleNet \
  --time=00-12:00 \
  bash generate_flis_pheno.sh
  # --mail-type=END --mail-user=naeem.iqbal@dfki.de --job-name=FreestyleNet \
