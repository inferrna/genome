import numpy as np
dm = 1
counters = "ijkzhiuy"
sums = "abcdef"
lneurons = ["lnr", "dnr"]
a = np.array([7, 1], dtype=np.uint32)
conns = (a[:-1]*a[1:]).sum()
samples = 5
#s.append("for({0}=0; {0}<{1}; {0}++)".format(counters[dm], str(len(a)))+"{ //\"Layers\" loop")        #Layers loop
currner = 0
nextner = 0
currcon = 0
n = 0
#nextner += a[n]
#s.append(dm*"\t"+"sums[{1}][{0}]+=xxx;".format(counters[dm], dm-1))
s = []
s.append("__kernel void runnet(__global float *gneurons, __global float *ggconns, __global float *gtargets, __global float *results){")
s.append(dm*"\t"+"float dnr[{0}];".format(a[1]))
s.append(dm*"\t"+"float lnr[{0}];".format(a[1]))
s.append(dm*"\t"+"float result = 0.0;")
s.append(dm*"\t"+"uint totconns = {0};".format(conns))
s.append(dm*"\t"+"uint gid = get_global_id(0);")
s.append(dm*"\t"+"float *gconns = ggconns+{0}*gid;".format(conns))
s.append(dm*"\t"+"for(uint {0}=0; {0}<{1}*{2}; {0}+={2})".format("n_sam", samples, a[0])+"{ //\"Samples\" loop ") #"Samples" loop
dm+=1
s.append(dm*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm], str(a[n]))+"{ //\"Neurons\" loop "+str(n)) #"Neurons" loop
s.append((dm+1)*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm+1], str(a[n+1]))+"{ //\"Connections\" loop "+str(n)) #"Connections" loop
s.append((dm+2)*"\t"+"{6}[{1}] += gneurons[{2}+{3}]*gconns[{3}*{5}+{1}]".format(\
                                                                               nextner, counters[dm+1],\
                                                                               "n_sam", counters[dm],\
                                                                               currcon, str(a[n+1]), 
                                                                               lneurons[n%2])+";\n"+(dm+1)*"\t"+"}") #"Connections" loop
s.append(dm*"\t"+"}") #"Connections" loop
currcon += a[n]*a[n+1]
for n in range(1, len(a)-1):
    nextner += a[n]
    #s.append(dm*"\t"+"sums[{1}][{0}]+=xxx;".format(counters[dm], dm-1))
    s.append(dm*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm], str(a[n]))+"{ //\"Neurons\" loop "+str(n)) #"Neurons" loop
    if(a[n+1]>1):
        s.append((dm+1)*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)"\
                             .format(counters[dm+1], str(a[n+1]))\
                             +"{ //\"Connections\" loop "+str(n)) #"Connections" loop
        nrcnt = counters[dm+1]
    else: nrcnt = 0
    s.append((dm+2)*"\t"+"{6}[{1}] += {7}[{3}]*gconns[{4}+{3}*{5}+{1}]".format(\
                                                                                   nextner, nrcnt,\
                                                                                   currner, counters[dm],\
                                                                                   currcon, str(a[n+1]),\
                                                                                   lneurons[n%2], lneurons[(n+1)%2])+";")
    if(a[n+1]>1):
        s.append((dm+1)*"\t"+"}") #"Connections" loop
    s.append((dm+1)*"\t"+"{0}[{1}] = 0.0;".format(lneurons[(n+1)%2], counters[dm]))
    s.append(dm*"\t"+"}") #"Neurons" loop
    currcon += a[n]*a[n+1]
    currner += a[n]
    #dm+=1
s.append((dm)*"\t"+"{0}[0] = 0.0;".format(lneurons[n%2])) #"Samples" loop
s.append((dm)*"\t"+"result += fabs(gtargets[{0}] - {1}[0]);".format("n_sam", lneurons[n%2])) #"Samples" loop
s.append((dm-1)*"\t"+"}") #"Samples" loop
s.append("results[gid] = result;\n}") #Kernel end
print("\n".join(s))
