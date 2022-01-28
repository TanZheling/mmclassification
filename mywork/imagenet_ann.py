import os

pathfile='/run/determined/workdir/datasets/imagenet/val_images.txt'
annfile='/run/determined/workdir/datasets/imagenet/meta/val.txt'
output='/run/determined/workdir/datasets/imagenet/val.txt'
pathlines=[]
anns=[]
f=open(output,"w")
with open(annfile) as ann:
    anns=ann.readlines()
with open(pathfile) as path:
    pathlines = path.readlines()
for pathline in pathlines:
    thispathline = pathline.split('/')
    #print(thispathline)
    imgname = thispathline[-1].rstrip()
    for ann in anns:
        ann = ann.split(' ')
        #print(ann)
        if ann[0]==imgname:
            f.write(pathline.rstrip()+" "+ann[1].rstrip()+"\n")
            #print(pathline.rstrip()+" "+ann[1].rstrip()+"\n")
