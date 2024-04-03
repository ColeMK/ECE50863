quality_levels = [1,2,4]
upcoming_levels = [[1,2,4],[1,3,5],[1,6,9],[2,4,6]]
window = 5
total_items = 0
whole_dict = {}
seconds_per_chunk = 1
buffer_fill = 0
buffer_max =30
Ct = 1
w1 = 4
w2 = 8
w3 = 3
def brute_force_func(partial_dict, depth):
    global total_items
    depth += 1
    if depth == 4:
        return
    
    for k,v in partial_dict.items():
        total_items += 1
        partial_dict[k] = {value: None for value in range(len(upcoming_levels[depth]))}
        brute_force_func(partial_dict[k], depth)
    


    
for i in range(len(quality_levels)): #because we want the index
    total_items += 1
    depth = 0
    whole_dict[i] = {value: None for value in range(len(upcoming_levels[depth]))}
    brute_force_func(whole_dict[i], depth)
print(total_items)

def update_bk(buffer_fill, encoded_qual, Ct, seconds_per_chunk, buffer_max):
    dt = (max(buffer_fill - ((encoded_qual)/Ct),0) + seconds_per_chunk - buffer_max)
    b1 = (max(buffer_fill - ((encoded_qual)/Ct),0) + seconds_per_chunk - dt)
    return b1

def buffer_qual(Ct, encoded_qual, bk):
    #print(max(encoded_qual/Ct-bk,0))
    return(max(encoded_qual/Ct-bk,0))

def dict_rec(dictionary,val,prev, var, start, Ct, buffer_max, bk, buffer_comp):
    for k,v in dictionary.items():
        if v != None:
            if start == 0:
                dict_rec(v, val+k, k, 0, 1, Ct, buffer_max,update_bk(buffer_fill=bk, encoded_qual=quality_levels[k], Ct=Ct, 
                                                    seconds_per_chunk=seconds_per_chunk,buffer_max=buffer_max), 
                                                    buffer_comp=buffer_comp + buffer_qual(Ct,quality_levels[k], bk))
            else:
                #print(f"Key: {k}, var {var + abs(k-prev)}, start {start}")
                dict_rec(v, val+k, k, var + abs(k-prev), start+1, Ct, buffer_max, update_bk(buffer_fill=bk, encoded_qual=upcoming_levels[start-1][k], Ct=Ct, 
                                                    seconds_per_chunk=seconds_per_chunk,buffer_max=buffer_max), 
                                                    buffer_comp + buffer_qual(Ct,upcoming_levels[start-1][k], bk))
                
        
        else:
            val_score = val + k
            var_score = var + abs(k-prev)
            buffer_score = buffer_comp + buffer_qual(Ct,upcoming_levels[start-1][k], bk)
            dictionary[k] = [val_score, var_score, buffer_score, w1*val_score-w2*var_score-w3*buffer_score]
curr_max = -9999
def get_max_rate(dictionary, start, final_dict,top):
    global curr_max
    for k,v in dictionary.items():
        if start == 0:
            top = k
        #print()
        if type(v) == list:
            if v[3] > curr_max:
                curr_max = v[3]
                final_dict['final'] = top
                print(top)
        else:
            get_max_rate(v, start+1, final_dict, top)       
# dt = (max(buffer_fill - ((encoded_qual)/Ct),0) + seconds_per_chunk - buffer_max)
# b1 = (max(buffer_fill - ((encoded_qual)/Ct),0) + seconds_per_chunk - dt)

dict_rec(dictionary=whole_dict, val=0,prev=0,var=0,start=0, Ct = Ct, buffer_max=buffer_max, bk = buffer_fill, buffer_comp=0)
final_dict = {'final': None}
top = -1
start = 0
get_max_rate(dictionary=whole_dict, start = start, final_dict=final_dict, top = -1)
print(whole_dict)
print(final_dict, curr_max)  
