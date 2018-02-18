# UnitThree
DescriptiveStats

# Lesson 1
import pandas as pd
import numpy as np

my_array=np.array([['Greg','Marcia','Peter','Jan','Bob','Cindy','Oliver'],[14,12,11,10,8,6,8]])
df = pd.DataFrame(my_array)
df.index=['Name', 'age']
df
0	1	2	3	4	5	6
Name	Greg	Marcia	Peter	Jan	Bob	Cindy	Oliver
age	14	12	11	10	8	6	8

​

names = ['Greg','Marcia','Peter','Jan','Bob','Cindy','Oliver']
​
bradybunch = pd.DataFrame(index=names)
​
bradybunch['age']=[14,12,11,10,8,7,1]
bradybunch
age
Greg	14
Marcia	12
Peter	11
Jan	10
Bob	8
Cindy	7
Oliver	1

np.mean(bradybunch['age'])
#np.median(bradybunch['age'])
#(values,counts)=np.unique(bradybunch['age'],return_counts=True)
#ind = np.argmax(counts)
#values[ind]


np.median(bradybunch['age'])


(values,counts)=np.unique(bradybunch['age'],return_counts=True)
ind = np.argmax(counts)
values[ind]


bradybunch['age'].var()


np.std(bradybunch['age'],ddof=1)


np.std(bradybunch['age'] ,ddof=1) / np.sqrt(len(bradybunch['age']))



# Lesson 3
import numpy as np
import matplotlib.pyplot as plt
var1 = np.random.normal(5,.5, 100)
var2 = np.random.normal(10,1, 100)
var3 = var1+var2
​
mean = np.mean(var3)
sd = np.std(var3)
​
plt.hist(var3)
plt.axvline(x=mean,color='black')
plt.axvline(x=mean+sd, color='red')
plt.axvline(x=mean-sd, color='red')
plt.show()
<matplotlib.figure.Figure at 0x1065c1550>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
Type Markdown and LaTeX:  α2α2 

binomial = np.random.binomial(20, 0.5, 1000)
binomial.sort()
plt.hist(binomial, color='c')
plt.axvline(binomial.mean(), color='b', linestyle='solid', linewidth=2)
plt.axvline(binomial.mean() + binomial.std(), color='b', linestyle='dashed', linewidth=2)
plt.axvline(binomial.mean()-binomial.std(), color='b', linestyle='dashed', linewidth=2)
<matplotlib.lines.Line2D at 0x112b12ef0>


gamma = np.random.gamma(10,.5, 2000)
plt.hist(gamma, bins=20, color='c')
plt.axvline(gamma.mean(), color='b', linestyle='solid', linewidth=2)
plt.axvline(gamma.mean() + gamma.std(), color='b', linestyle='dashed', linewidth=2)
plt.axvline(gamma.mean()-gamma.std(), color='b', linestyle='dashed', linewidth=2)
<matplotlib.lines.Line2D at 0x112f21ba8>


rayleigh = np.random.rayleigh([.5, 2000])
plt.hist(rayleigh, color='c')
plt.axvline(rayleigh.mean(), color='b', linestyle='solid', linewidth=2)
plt.axvline(rayleigh.mean() + rayleigh.std(), color='b', linestyle='dashed', linewidth=2)
plt.axvline(rayleigh.mean()-rayleigh.std(), color='b', linestyle='dashed', linewidth=2)
<matplotlib.lines.Line2D at 0x112ba7f28>


poisson = np.random.poisson(3, 1000)
plt.hist(poisson, color='c')
plt.axvline(poisson.mean(), color='b', linestyle='solid', linewidth=2)
plt.axvline(poisson.mean() + poisson.std(), color='b', linestyle='dashed', linewidth=2)
plt.axvline(poisson.mean()-poisson.std(), color='b', linestyle='dashed', linewidth=2)
<matplotlib.lines.Line2D at 0x112774828>


negative_bionomial = np.random.negative_binomial(5, .4,[100])
plt.hist(negative_bionomial, color='c')
plt.axvline(negative_bionomial.mean(), color='b', linestyle='solid', linewidth=2)
plt.axvline(negative_bionomial.mean() + negative_bionomial.std(), color='b', linestyle='dashed', linewidth=2)
plt.axvline(negative_bionomial.mean()-negative_bionomial.std(), color='b', linestyle='dashed', linewidth=2)
<matplotlib.lines.Line2D at 0x112a1b8d0>


standard_normal=np.random.standard_normal([1000])
plt.hist(standard_normal, color='c')
plt.axvline(standard_normal.mean(), color='b', linestyle='solid', linewidth=2)
plt.axvline(standard_normal.mean() + standard_normal.std(), color='b', linestyle='dashed', linewidth=2)
plt.axvline(standard_normal.mean()-standard_normal.std(), color='b', linestyle='dashed', linewidth=2)
<matplotlib.lines.Line2D at 0x113541908>


import numpy as np
import matplotlib.pyplot as plt
var1 = np.random.normal(5,.5, 100)
var2 = np.random.normal(10,1, 100)
var3 = var1+var2
​
mean = np.mean(var3)
sd = np.std(var3)
​
plt.hist(var3)
plt.axvline(x=mean,color='black')
plt.axvline(x=mean+sd, color='red')
plt.axvline(x=mean-sd, color='red')
plt.show()


pop1 = np.random.binomial(10, 0.2, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 
​
sample1 = np.random.choice(pop1, 1000, replace=True)
sample2 = np.random.choice(pop2, 1000, replace=True)
​
print(sample1.mean())
print(sample1.std())
print(sample2.mean())
print(sample2.std())
​
difference=sample2.mean( ) -sample1.mean()
print(difference)
​
plt.hist(sample1, alpha=0.5, label='sample 1') 
plt.hist(sample2, alpha=0.5, label='sample 2') 
plt.legend(loc='upper right') 
plt.show()


pop1 = np.random.binomial(10, 0.3, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 
​
sample1 = np.random.choice(pop1, 100, replace=True)
sample2 = np.random.choice(pop2, 100, replace=True)
​
from scipy.stats import ttest_ind
print(ttest_ind(sample2, sample1, equal_var=False))
Ttest_indResult(statistic=11.083987229905155, pvalue=3.347454295731432e-22)

​
