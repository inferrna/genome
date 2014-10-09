import numpy as np
dm = 1
counters = "ijkzhiuy"
sums = "abcdef"
lneurons = ["lnr", "dnr"]
a = np.array([5, 3, 2, 1], dtype=np.uint32)
s = []
s.append("float sum = 0.0;")
s.append("float sums[{0}];".format(a.sum()))
#s.append("for({0}=0; {0}<{1}; {0}++)".format(counters[dm], str(len(a)))+"{ //\"Layers\" loop")        #Layers loop
currner = 0
nextner = 0
currcon = 0
n = 0
#nextner += a[n]
#s.append(dm*"\t"+"sums[{1}][{0}]+=xxx;".format(counters[dm], dm-1))
s.append(dm*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm], str(a[n]))+"{ //\"Neurons\" loop "+str(n)) #"Neurons" loop
s.append((dm+1)*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm+1], str(a[n+1]))+"{ //\"Connections\" loop "+str(n)) #"Connections" loop
s.append((dm+2)*"\t"+"{6}[{1}] += gneurons[{3}]*conns[{3}*{5}+{1}]".format(\
                                                                               nextner, counters[dm+1],\
                                                                               currner, counters[dm],\
                                                                               currcon, str(a[n+1]), 
                                                                               lneurons[n%2])+";\n"+(dm+1)*"\t"+"}") #"Connections" loop
s.append(dm*"\t"+"}") #"Connections" loop
currcon += a[n]*a[n+1]
for n in range(1, len(a)-1):
    nextner += a[n]
    #s.append(dm*"\t"+"sums[{1}][{0}]+=xxx;".format(counters[dm], dm-1))
    s.append(dm*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm], str(a[n]))+"{ //\"Neurons\" loop "+str(n)) #"Neurons" loop
    s.append((dm+1)*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm+1], str(a[n+1]))+"{ //\"Connections\" loop "+str(n)) #"Connections" loop
    s.append((dm+2)*"\t"+"{6}[{1}] += {7}[{3}]*conns[{4}+{3}*{5}+{1}]".format(\
                                                                                   nextner, counters[dm+1],\
                                                                                   currner, counters[dm],\
                                                                                   currcon, str(a[n+1]),\
                                                                                   lneurons[n%2], lneurons[(n+1)%2])+\
                                                                                   ";\n"+(dm+1)*"\t"+"}") #"Connections" loop
    s.append((dm+1)*"\t"+"{0}[{1}] = 0.0;".format(lneurons[(n+1)%2], counters[dm])) #"Connections" loop
    s.append(dm*"\t"+"}") #"Connections" loop
    currcon += a[n]*a[n+1]
    currner += a[n]
    #dm+=1
print("\n".join(s))
