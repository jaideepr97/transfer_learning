filenames = ['test-pos.txt', 'test-neg.txt', 'train-pos.txt', 'train-neg.txt']
with open('consolidated.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
