inp='/run/determined/workdir/datasets/imagenet/val.txt'
output='/run/determined/workdir/scratch/mmclassification/mywork/val.txt'
f=open(output,"w")
with open(inp) as file:
    lines = file.readlines()
    for line in lines:
        line=line.split('/',1)
        print(line)
        f.write(line[1])