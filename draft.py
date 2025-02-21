def modify_mutable(lst):
    lst = [2,2,2]
    print("Inside function:", lst)

my_list = [1, 2, 3]
modify_mutable(my_list)
print("Outside function:", my_list)