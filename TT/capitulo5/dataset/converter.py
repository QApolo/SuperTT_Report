# get the tokens
tokens_file = open('latex_vocab.txt', 'r')
tokens = dict()
count = 0

for line in tokens_file:
    count += 1
    tokens[line.strip()] = count

tokens_file.close()

expressions_file = open('expressions.txt', 'r')
expressions_norm_file = open('expressions.norm.lst', 'r')
expressions_norm = expressions_norm_file.read().split('\n')
tokenized = open('tokenized.csv', 'w')
MAX_LENGHT = 103
i = 0

for line in expressions_file:

	if 'Object' in expressions_norm[i] or '\\div' in expressions_norm[i]:
		i += 1
		continue

	expressions_split = line.split('$')
	expression = expressions_norm[i].split(' ')
	sequence = '1000'

	for e in expression:
			sequence += ' ' + str(tokens[e])

	sequence += ' 1001'

	for x in range(0, MAX_LENGHT - len(expression)):
		sequence += ' 0'

	tokenized.write(expressions_split[0] + ',' + sequence + '\n')
	i += 1
