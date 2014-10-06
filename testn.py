dm = 0
counters = "ijkzhiuy"
a = [3, 2, 1]
s = ["for({0}=0; {0}<{1}; {0}++)".format(counters[dm], str(len(a)))+"{"]
for n in a:
    dm+=1
    s.append(dm*"\t"+"for({0}=0; {0}<{1}; {0}++)".format(counters[dm], str(n))+"{")
print("\n".join(s))
