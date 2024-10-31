# Batch process commands

run both commands on your local machine
/temp/img must exist first

## Upload files for processing

```bash
scp -P <port> -r /temp/img/input root@<ip>:/workspace/img
```

rsync is better
```bash
rsync -avz -e "ssh -p <port>" --ignore-existing --no-perms --no-owner --no-group /mnt/c/temp/img/input/ root@<ip>:/workspace/img/input
```

## Download files after a run

```bash
scp -P <port> -r root@<ip>:/workspace/img/output /temp/img
```

if you wanted just the files from a batch then 
```bash
scp -P <port> -r root@<ip>:/workspace/img/output/batch1 /temp/img/output
```

rsync will be a better option...
```bash
rsync -avz -e "ssh -p <port>" --ignore-existing --no-perms root@<ip>:/workspace/img/output/ /mnt/c/temp/img/output
```
