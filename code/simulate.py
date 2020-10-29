def create_set(c,N):
    if not type(c) is complex:
        raise TypeError("Only complex numbers allowed")
    z = [0]
    for i in range(N):
        z.append(z[i]**2 + c)
    return z

print(create_set(complex(1,3),10))

