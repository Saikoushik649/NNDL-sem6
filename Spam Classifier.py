import math

free = 0
win = 1
often = 1


h1 = free * 0.5
h1 = h1 + win * (-0.2)
h1 = h1 + often * 0.3
h1 = max(0, h1)

h2 = free * 0.4
h2 = h2 + win * 0.1
h2 = h2 + often * (-0.5)
h2 = max(0, h2)


out = (h1 * 0.7) + (h2 * 0.2)
spam = 1 / (1 + math.exp(-out))

print("n1 =", h1)
print("n2 =", h2)
print("Raw output =", out)
print("Spam probability =", spam)


OUTPUT:
n1 = 0.09999999999999998
n2 = 0
Raw output = 0.06999999999999998
Spam probability = 0.5174928576663897
