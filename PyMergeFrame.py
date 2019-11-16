#coding:utf-8
#根据同一路径下的VideoFramePaths.txt调用ffmpeg按顺序合成视频
import os
import shutil
import time
Fps=26
t_start=time.perf_counter()
fid=open('VideoFramePaths.txt','r')
Text=fid.read()
fid.close()
Text=Text.replace('\r\n','\n')
srcList=Text.split('\n')

#maxL=0
#for item in srcList:
#    L=len(item)
#    if L>maxL:
#        maxL=L
#ffmpeg不支持太长的
maxL=8
if os.path.exists('Tmp'):
    shutil.rmtree('Tmp')
os.mkdir('Tmp')
N=len(srcList)
k=0
i=0
for item in srcList:
    i=i+1
    if os.path.exists(item):
        ExName=item.split('.')[-1]
        k=k+1
        dstPath=('./Tmp/Frame_%.'+str(maxL)+'d.'+ExName)%(k)
        shutil.copyfile(item,dstPath)
    print('\r','    File done:%d of %d'%(i,N), end = '',flush=True)
t_end=time.perf_counter()
print('……These files have been copied in %.17gs'%(t_end-t_start))

print('---------------------------------------------------')
t_start=time.perf_counter()
command='ffmpeg -i ./Tmp/Frame_%'+str(maxL)+'d.'+ExName+' -r '+str(Fps)+' PyMergeDst.mp4'
print(command)
res=os.system(command)
t_end=time.perf_counter()
print('---------------------------------------------------')
print('These images have been merged in %.17gs'%(t_end-t_start))
shutil.rmtree('Tmp')
