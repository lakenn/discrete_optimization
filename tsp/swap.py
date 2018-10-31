def swap(solution, x, y):
    return solution[:x] + solution[x:y + 1][::-1] + solution[y + 1:]

a =  [ 1,5,9,10,6,3,4,8,11,7,2]
route = [1,5,9,10,6,3,4,8,11,7,2]
for i in range(1, len(route) -1):
    for k in range(i+1, len(route)):
        print(route[:i])
        print(route[i:k+1])
        print("-----")
        print(swap(route, i, k))