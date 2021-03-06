import numpy as np
from sys import stdout, argv
from os import system
import time

#Possible arguments are:
#
# -r : randomize the starting board
# -p : print out the board on the console at each iteration
# -i <int value> : runs the board over <value> iterations
# -s : stops the board if there are repeats

#Set up with -b <int>
board_size = 8

#Boards are generated ahead of time to avoid allocating more memory
#boardmod0 set to random with -r
boardmod0 = np.full((8,8), False)
boardmod1 = np.full((8,8), True)

#Set to a value with -i <value>
total_iterations = 100

#Set to True with -p
will_print_board = False

#Set to True with -s
stop_after_repeat = False

padding = 1

def print_board(counter):
    """ 
    Prints the board on the console.

    Parameters:
    counter: the counter telling which board to print

    Returns: 
    None
    
    Side-effects: 
    Prints the board on the screen
    Waits a tenth of a second afterwards
    """
    
    system('clear')
    board = boardmod0 if counter % 2 == 0 else boardmod1
    for row in board:
            for element in row:
                if (element):
                    stdout.write("X")
                else:
                    stdout.write(".")
            stdout.write("\n")
    time.sleep(.1)

def check_neighbors(board, row, column):
    """
    Checks how many neighbors a cell has.

    Parameters:
    board: The board to check.
    row: The row of the current cell
    column: The column of the current cell

    Returns:
    Integer number of live (True) neighbors a cell has
    """

    #This way we know what the "neighbors" of the cell are,
    #in case of being on edge we set the neighbors to be the other side
    #of the grid
    check_left = column-1
    check_right = column+1
    check_up = row - 1
    check_down = row + 1
    
    num_neighbors = 0

    #TODO: Make this code shorter
    if board[check_up][check_left] == True:
        num_neighbors+=1
    if board[check_up][column] == True:
        num_neighbors+=1
    if board[check_up][check_right] == True:
        num_neighbors+=1
    if board[row][check_left] == True:
        num_neighbors+=1
    if board[row][check_right] == True:
        num_neighbors+=1
    if board[check_down][check_left] == True:
        num_neighbors+=1
    if board[check_down][column] == True:
        num_neighbors+=1
    if board[check_down][check_right] == True:
        num_neighbors+=1

    return num_neighbors
        
def update_board(counter):
    """
    Takes one board and uses it to overwrite the other board
    with the cells obtained from iterating the rules once.

    Parameters:
    counter: The counter determining what board to update

    Returns:
    None

    Side-effects:
    Updates one board based on the value of counter
    """
    
    board = boardmod0 if counter % 2 == 0 else boardmod1
    board_to_update = boardmod1 if counter % 2 == 0 else boardmod0
    #board_to_update = np.pad(board_to_update.reshape(10,10), (1,1), mode='wrap')
    for c in range(padding, board_size+padding):
        board_to_update[0][c] = board[board_size+padding][c]
        board_to_update[board_size+padding][c] = board[0][c]
    for r in range(padding, board_size+padding):
        board_to_update[r][0] = board[r][board_size+padding]
        board_to_update[r][board_size+padding] = board[r][0]
    board_to_update[0][0]=board[board_size+padding][board_size+padding]
    board_to_update[0][board_size+2*padding-1]=board[board_size+padding][padding]
    board_to_update[board_size+2*padding-1][0]=board[padding][board_size+padding]
    board_to_update[board_size+2*padding-1][board_size+2*padding-1]=board[padding][padding]
        
    for row in range(padding, board_size+padding):
        for column in range(padding, board_size+padding):
            cell_state = board[row][column]
            num_neighbors = check_neighbors(board, row, column)
            if ((cell_state and not (num_neighbors == 2 or num_neighbors == 3))
                or (not cell_state and num_neighbors == 3)):
                board_to_update[row][column] = not board[row][column]
            else:
                board_to_update[row][column] = board[row][column]

def is_int(arg):
    """ Checks if the parameter is an integer """
    try:
        int(arg)
        return True
    except:
        return False

setboardmod0 = False
    
#argument parsing
if (len(argv) > 1):
    for arg in argv[1:]:
        if (arg == "-r"):
            setboardmod0=True
        if (arg == "-i"):
            try:
                next_arg = argv[argv.index(arg)+1]
                if (is_int(next_arg)):
                    total_iterations = int(next_arg)
                else:
                    stdout.write("Incorrect usage of argument -i. Please add an integer afterwards.\n")
                    quit()
            except:
                stdout.write("Incorrect usage of argument -i. Please add an integer afterwards.\n")
                quit()
        if (arg == "-b"):
            try:
                next_arg = argv[argv.index(arg)+1]
                if (is_int(next_arg)):
                    board_size = int(next_arg)
                else:
                    stdout.write("Incorrect usage of argument -i. Please add an integer afterwards.\n")
                    quit()
            except:
                stdout.write("Incorrect usage of argument -i. Please add an integer afterwards.\n")
                quit()
        if (arg == "-p"):
            will_print_board = True
        if (arg == "-s"):
            stop_after_repeat = True

counter = 0

#Play over a total_iterations number of iterations
#If -s is passed, stop if boards repeat
if setboardmod0:
    boardmod0 = np.random.choice(a=[False, True], size=(board_size+2*padding,board_size+2*padding), p=[.5,.5])

boardmod1 = np.full((board_size+2*padding, board_size+2*padding), False)

for i in range(total_iterations):
    if (stop_after_repeat):
        if np.array_equal(boardmod0,boardmod1):
            break
    if (will_print_board):
        print_board(counter)
    update_board(counter)
    counter += 1
