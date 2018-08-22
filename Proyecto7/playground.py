import numpy as np
c1 = np.array([4,2.5,4.5])
c0 = np.array([0,2,2.5])

d1 = np.array([2,3,5])
d2 = np.array([1,3,2])
d3 = np.array([6,2,4])
d4 = np.array([-1,1,3])

print "d1 - c0 ",np.linalg.norm(d1-c0)
print "d1 - c1 ",np.linalg.norm(d1-c1)
print
print "d2 - c0 ",np.linalg.norm(d2-c0)
print "d2 - c1 ",np.linalg.norm(d2-c1)
print
print "d3 - c0 ",np.linalg.norm(d3-c0)
print "d3 - c1 ",np.linalg.norm(d3-c1)
print
print "d4 - c0 ",np.linalg.norm(d4-c0)
print "d4 - c1 ",np.linalg.norm(d4-c1)

toSum = [d2,d4]
t = len(toSum)
xs = 0.
ys = 0.
zs = 0.
for d in toSum:
	xs+=d[0]
	ys+=d[1]
	zs+=d[2]

print "C ",(xs/t,ys/t,zs/t)