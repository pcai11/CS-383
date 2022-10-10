"""Test boards for Connect383

Place test boards in this module to help test your code.  Note that since connect383.GameState
stores board contents as a 0-based list of lists, these boards are reversed to they can be written
right side up here.

"""

boards = {}  # dictionary with label, test board key-value pairs

boards['test_1'] = reversed([  
    [  0,  0 ],                       
    [  1, -1 ],
    [  1, -1 ] 
])

boards['test_2'] = reversed([  
    [ 0,   0,  0, 0, 0 ],  
    [ 0,   0, 0,  0,  0 ],
    [  1,  -1,  1, -1,  1 ],
    [  1,   1, -1,  1, -1 ] 
])

boards['writeup_1'] = reversed([
    [  0,  0,  0,  0,  0,  0,  0 ],
    [  0,  0,  0,  0,  0,  0,  0 ],
    [  0,  0,  0,  0,  0,  0,  0 ],
    [  0,  0,  0, -1,  0,  0,  0 ],
    [  0,  0,  0,  1,  0,  0,  0 ],
    [  0,  1,  0, -1,  0,  1,  0 ]
])

boards['writeup_2'] = reversed([  
    [ -1,  1, -1, -1 ],                       
    [  1, -1,  1, -1 ],
    [  1, -2, -1,  1 ],
    [  1, -2,  1, -1 ] 
])

boards['empty'] = reversed([
    [  0,  0, 0 ],                       
    [  0, 0, 0 ],
    [  0, 0, 0] 
])  

boards['one_move'] = reversed([
    [  -1,  1, 0, -1 ],                       
    [  1, -1, 1, 1 ],
    [  -1, 1, 1, -1 ] 
])  

boards['one_column'] = reversed([
    [  -1,  1, 0, -1 ],                       
    [  1, -1, 0, 1 ],
    [  -1, 1, 0, -1 ] 
])  

boards['sanity1'] = reversed([
    [  1, 1, 0, 1 ],                       
    [  1, 1, 1, 1 ],
    [  1, 1, 1, 1 ] 
])

boards['sanity-1'] = reversed([
    [  0, 0, 0],                       
    [  -1, -1, -1],
    [  -1, -1, -1] 
])

