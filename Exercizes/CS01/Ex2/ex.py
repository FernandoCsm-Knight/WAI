points = [(2, 4), (4, 2)]

def F(w):
    return sum((w * x - y)**2 for x, y in points)

def dF(w):
    return sum(2 * (w * x - y) * x for x, y in points)

w = 0
eta = 0.01
for i in range(100):
    value = F(w)
    gradient = dF(w)
    w = w - eta * gradient
    print(f"interation {i}: w = {w}, F(w) = {value}, dF(w) = {gradient}")