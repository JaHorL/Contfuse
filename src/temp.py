import time
 
localtime = time.asctime( time.localtime(time.time()) )
print("本地时间为 :", localtime)