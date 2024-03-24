# list1 = [1,2,-1,4,5]
# indexes_of_minus_one = [index for index, value in enumerate(list1) if value == -1]

# if len(indexes_of_minus_one) > 0:
#     nack_index = indexes_of_minus_one[0]
# else: 
#     nack_index = len(list1)


# if nack_index != 0:
#     before = list1[nack_index-1]
# else:
#     before = -1

#shiftList = list1[cum_index:] + [-1]*cum_index
import ast
listString = '[]'
data = ast.literal_eval(listString)
print(data)