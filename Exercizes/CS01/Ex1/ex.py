s: str = input("Enter some string: ")
t: str = input("Enter some string to compare: ")

def changesStoTin(s: str, t: str) -> int: 
    cache = {}
    def recurse(m, n):
        num = 0

        if (m, n) in cache:
            return cache[(m, n)]
        
        if m == 0:
            num = n
        elif n == 0:
            num = m
        elif s[m - 1] == t[n - 1]:
            num = recurse(m - 1, n - 1)
        else:
            subs = 1 + recurse(m - 1, n - 1)
            inse = 1 + recurse(m, n - 1)
            dele = 1 + recurse(m - 1, n)
            num = min(subs, inse, dele)

        cache[(m, n)] = num
        return num
        
    return recurse(len(s), len(t))
        
print(f"The string \"{s}\" is at a distance of {changesStoTin(s*10, t*10)} characters from the string \"{t}\"")