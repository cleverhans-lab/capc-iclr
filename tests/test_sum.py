import unittest
import numpy as np
import os
from threading import Thread

PATH = '../gc-emp-test/'

def read_array(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(int(line.strip()))
    return np.array(data)


def write_array(filename, array):
    with open(filename, 'w') as writer:
        for element in array:
            writer.write(str(element) + '\n')
            
def clean_directory():
    if os.path.exists('{}party1.txt'.format(PATH)):
        os.remove('{}party1.txt'.format(PATH))
    if os.path.exists('{}party2.txt'.format(PATH)):
        os.remove('{}party2.txt'.format(PATH))
    if os.path.exists('{}output.txt'.format(PATH)):
        os.remove('{}output.txt'.format(PATH))
            
def run_command(party_filenames, output_filename = 'output.txt'):
    cmds = []
    for i, filename in enumerate(party_filenames):
        cmds.append('./{}bin/sum_histogram {} 12345 {} {}'.format(PATH, str(i+1), filename, output_filename))
    cmd = ' & '.join(cmds)
    thread = Thread(target = os.system, args = (cmd, ))
    thread.start()
    thread.join()

class TestArgmaxSumMethod(unittest.TestCase):
    
    def test_smallint(self):
        clean_directory()
        party1 = np.random.randint(-(2**8), 2**8, 32)
        party2 = np.random.randint(-(2**8), 2**8, 32)
        print('Party 1 vector: ', party1)
        print('Party 2 vector: ', party2)
        write_array('{}party1.txt'.format(PATH), party1)
        write_array('{}party2.txt'.format(PATH), party2)
        correct = np.argmax(party1+party2)
        run_command(['party1.txt', 'party2.txt'], 'output.txt')
        output = read_array('{}output.txt'.format(PATH)).item()
        self.assertEqual(correct, output)
        
    def test_mediumint(self):
        clean_directory()
        party1 = np.random.randint(-(2**24), 2**24, 32)
        party2 = np.random.randint(-(2**24), 2**24, 32)
        print('Party 1 vector: ', party1)
        print('Party 2 vector: ', party2)
        write_array('{}party1.txt'.format(PATH), party1)
        write_array('{}party2.txt'.format(PATH), party2)
        correct = np.argmax(party1+party2)
        run_command(['party1.txt', 'party2.txt'], 'output.txt')
        output = read_array('{}output.txt'.format(PATH)).item()
        self.assertEqual(correct, output)

    def test_largeint(self):
        clean_directory()
        party1 = np.random.randint(-(2**40), 2**40, 32)
        party2 = np.random.randint(-(2**40), 2**40, 32)
        print('Party 1 vector: ', party1)
        print('Party 2 vector: ', party2)
        write_array('{}party1.txt'.format(PATH), party1)
        write_array('{}party2.txt'.format(PATH), party2)
        correct = np.argmax(party1+party2)
        run_command(['party1.txt', 'party2.txt'], 'output.txt')
        output = read_array('{}output.txt'.format(PATH)).item()
        self.assertEqual(correct, output)
    
    def test_short(self):
        clean_directory()
        party1 = np.random.randint(-(2**24), 2**24, 16)
        party2 = np.random.randint(-(2**24), 2**24, 16)
        print('Party 1 vector: ', party1)
        print('Party 2 vector: ', party2)
        write_array('{}party1.txt'.format(PATH), party1)
        write_array('{}party2.txt'.format(PATH), party2)
        correct = np.argmax(party1+party2)
        run_command(['party1.txt', 'party2.txt'], 'output.txt')
        output = read_array('{}output.txt'.format(PATH)).item()
        self.assertEqual(correct, output)
    
    def test_long(self):
        clean_directory()
        party1 = np.random.randint(-(2**24), 2**24, 128)
        party2 = np.random.randint(-(2**24), 2**24, 128)
        print('Party 1 vector: ', party1)
        print('Party 2 vector: ', party2)
        write_array('{}party1.txt'.format(PATH), party1)
        write_array('{}party2.txt'.format(PATH), party2)
        correct = np.argmax(party1+party2)
        run_command(['party1.txt', 'party2.txt'], 'output.txt')
        output = read_array('{}output.txt'.format(PATH)).item()
        self.assertEqual(correct, output)

if __name__ == '__main__':
    unittest.main()

