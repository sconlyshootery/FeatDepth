import os

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

if __name__ == "__main__":

    filenames = readlines("train_files.txt")
    unnecessary = readlines("unused.txt")

    unnecessary = map(int, unnecessary)
    unnecessary = sorted(unnecessary)
    unnecessary.reverse()
    for i in unnecessary:
        del filenames[i]
    os.remove('./train_files.txt')
    with open('./train_files.txt', 'w') as f:
        for i in range(len(filenames)):
            f.write('{}\n'.format(filenames[i]))
    print('done')