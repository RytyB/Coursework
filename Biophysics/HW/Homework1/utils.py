import numpy as np

############# Utils for Problem 2 ###############
def countLongestRun(flipSequence, verbose=False):
    '''
    Takes as input an iterable sequence of 0s and 1s
    - Should work more generally than that, but not tested
    Output: ( longest run length, value with longest run ) as tuple
    - A longestVal of 0.5 means there were an equal number of 0 and 1
    '''

    longest = 1
    current = 1
    tie = True
    for i in range( len(flipSequence)-1 ):
        if ( flipSequence[i] == flipSequence[i+1] ) or ( i == len(flipSequence)-1 ):
            current += 1
        else:
            current = 1
        
        if current > longest:
            tie = False
            longest = current
            longestVal = flipSequence[i]
        elif current == longest:
            tie == True

    if tie:
        longestVal = .5
        
    if verbose:
        print('Length of longest:\t' + str(longest))
        print('Value of longest:\t' + str(longestVal))

    return longest, longestVal

def flipNCoins(N):
    '''
    Input is number of coins to flip. Returns iterable sequence of 0 and 1.
    '''
    seq = []
    for flip in range(0, N):
        hort = np.random.randint(0,2)
        seq.append( hort )
        
    return seq


########### Utilities for Problem 3 #############
def bin2d(data, N=15, M=15):
    '''
    Input:
    - data is the original .npz file
    - N is the number of vertical compartments
    - M is the number of horizontal bins
    Output:
    counts (NxM)
    - counts is a matrix where each element is the number of counts in the corresponding cell
    - for the data in the problem, the matrix elements correspond as follows
    | 02  12  22 ... |
    | 01  11  21 ... |
    | 00  10  20 ... |
    '''

    counts = np.zeros( (N,M) )

    epsilon = 1e-6

    # Here we add epsilon to avoid equality with the upper bounds
    xmin = np.min(data['all'][:,0]) - epsilon
    xmax = np.max(data['all'][:,0]) + epsilon
    xBinWidth = (xmax-xmin) / (N)  # Add two to put outer bounds on the data
    ymin = np.min(data['all'][:,1]) - epsilon
    ymax = np.max(data['all'][:,1]) + epsilon
    yBinWidth = (ymax-ymin) / (M)

    
    # Here we add epsilon so that python will populate a value at xmax and ymax
    xBound = np.arange(xmin, xmax + epsilon, xBinWidth)
    yBound = np.arange(ymin, ymax + epsilon, yBinWidth)

    # Iterate through every data point to find which bin it belongs in
    for item in data['all']:

        # Iterate through every bin and check if data point belongs there
        for i in range(0, N):
            for j in range(0, M):

                if (item[0] > xBound[i]) and (item[0] <= xBound[i+1]):
                    if (item[1] > yBound[j]) and (item[1] <= yBound[j+1]):
                        counts[i,j] += 1

    return counts

def intBin(counts):
    '''
    Input: 
    - a counts matrix from bin2d funtion
    Output:
    - x_ax, y_ax: a functional tuple that plt.bar will actually accept
    NOTE: why are there no good librarys for histograms? How has the open source community not figured this out by now?
    '''

    maxVal = np.max( counts.flatten() )

    val = np.arange(0, maxVal + 1, 1)
    freq = np.zeros( np.shape(val) )

    for item in counts.flatten():
        freq[int(item)] += 1

    return val, freq

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # This code only exists to test the functions

    # testSeq = [1,0,1,1,0,0,0,0,0,0,0,0,0]         # longest sequence is nine
    # length, val = countLongestRun(testSeq, verbose=True)

    x = np.load('londonIncidents.npz')

    cellCounts = bin2d(x)
    x_ax, y_ax = intBin(cellCounts)

    print()
    print(cellCounts)
    print()
    print(x_ax)
    print()
    print(y_ax)

    fig, ax = plt.subplots( figsize=(12,6) )

    ax.bar(x_ax, y_ax, width = .1, color='black', align='center')

    ax.set_ylabel('Frequency')
    ax.set_xlabel('Number of cancer cases in a cell')

    plt.show()

    