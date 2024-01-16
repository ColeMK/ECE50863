import heapq

with open("Config/graph_3.txt", "r") as file:
    data = file.read()
    dataList = data.split('\n')
    if dataList[-1] == '':
        dataList.pop(-1)
    totSwitch = int(dataList[0])
    #graphDict = dict.fromkeys(range(totSwitch), dict())
    graphDict = {}
    for x in range(totSwitch):
        graphDict[x] = {}
    for i,value in enumerate(dataList):
        if i != 0:
            valueList = value.split(' ')
            valueList = [int(item) for item in valueList]
            print(valueList)
            for j,data in enumerate(valueList):
                if j !=2:
                    if j == 0:
                        graphDict[valueList[0]][valueList[1]] = valueList[2]
                        print(valueList[0])
                    elif j == 1:
                        graphDict[valueList[1]][valueList[0]] = valueList[2]

def dijkstra(graphDict, id):
    distances = {}
    paths = {}
    for i in graphDict:
        distances[i] = 9999
        paths[i] = []
    distances[id] = 0
    paths[id] = [id] #Shortest path to itself is itself
    pq = [(0,id)]
    
    while pq:
        cd, cn = heapq.heappop(pq) #current distance and current node
        jumpFlag = 0
        if cd <= distances[cn]: # may need to change
            for neighbor, distance in graphDict[cn].items():
                newDistance = cd + distance
                if newDistance < distances[neighbor]:
                    distances[neighbor] = newDistance
                    paths[neighbor] = paths[cn] + [neighbor]
                    heapq.heappush(pq, (newDistance,neighbor))
    
    return distances,paths

def formatRoutingTable(routing):
    formattedRoute = []
    for k,v in routing.items():
        for i,j in v.items():
            formattedRoute.append([k,i,j[0],j[1]])
    return formattedRoute

def generateRoutingTable(graphDict):
    completeRouting = {}
    for i in graphDict:
        completeRouting[i] = {}
    for id in graphDict:
        routingtable,shortestpaths = dijkstra(graphDict, id)
        for k,v in shortestpaths.items():
            if len(shortestpaths[k]) == 1: #format is key1 (start), key2 (destination), [next hop, tot distance]
                completeRouting[id][k] = [shortestpaths[k][0], routingtable[k]] 
            elif len(shortestpaths[k]) > 1:
                completeRouting[id][k] = [shortestpaths[k][1], routingtable[k]]
    for k,v in graphDict.items():
        for key,value in completeRouting.items():
            if k not in completeRouting[key]:
                completeRouting[key][k] = [-1,9999]
    return completeRouting

print(graphDict)
completeRouting = generateRoutingTable(graphDict)
controllerRouting = formatRoutingTable(completeRouting)
print(completeRouting)

print(controllerRouting)
