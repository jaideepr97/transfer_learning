import gensim
# import fasttext
import pandas as pd
import nltk
import re
import os
import pickle	
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

main_path = "/home/jaideeprao/Desktop/transfer_learning"
#os.join.path(main_path,'finalTry')
#os.chdir(os.path.join(main_path,'transfer_learning'))

import fast
from newtransfer_finaltest import classify
from fastText.shell import returns
#import shell
from subprocess import Popen, PIPE

from sklearn.neural_network import MLPClassifier

# nltk.download('stopwords')
#Main_Path = "/home/jaideeprao/Desktop/this/"
#os.chdir(Main_Path)
df=pd.read_csv('movie_reviews_data/test-pos.csv'); 

stop_words = set(stopwords.words('english'))
stop_words.add("I")
x=df['Reviews'].values.tolist()
tok_corp = [nltk.word_tokenize(sent) for sent in x]
model = gensim.models.Word2Vec(tok_corp, min_count=1)

filtered_sentence = []
returned_words = []
returned_prob = []
returned = []
obb = []
new_clust = []
new_obb = []
new_sorted_obb = []
new_collected_obb = []
new_averaged_obb = []
fasttext_return = []
new_fasttext_return = []
shell_return = []
new_shell_return = []
final_list = []
present = []
maybe_present = []

def preprocessing(testvar):
	word_tokens = word_tokenize(testvar)
	for w in word_tokens:
		if w not in stop_words:
			filtered_sentence.append(w)
   
   


def backend(strr):
	length = len(strr.split())
	fasttext_return = fast.return_list(strr)
	n = len(fasttext_return)
	i = 0
	while i < n:
		m = len(fasttext_return[i])
		j = 0
		while j < m:
			new_fasttext_return.append(fasttext_return[i][j])
			j = j + 1
		i = i + 1
	print("fasttext returned list:")
	print (new_fasttext_return)
	print(len(new_fasttext_return))

	print('\n')
	print('\n')
	j = 0
	while j < 5:
		p = Popen(['python cmd.py'], stdin=PIPE, shell=True)
		p.communicate(strr.encode())
		j = j + 1


	with open('parrot.pkl', 'rb') as f:
		while 1:
			try:
				data = pickle.load(f, encoding='latin1')
				obb.append(data)
			except EOFError:
				break	
     
	print(obb)		
	n = len(obb)
# print(n)
	i = 0
	while i < n:
		m = len(obb[i])
	# print(m)
		j = 0
		while j < m:
			o = len(obb[i][j])
		# print(o)
			k = 0
			while k < o:
				new_obb.append(obb[i][j][k])
				k = k + 1
			j = j + 1 
		i = i + 1	

	print('\n')
	new_sorted_obb = sorted(new_obb)

	print('\n')

	new_collected_obb = [(uk,sum([vv for kk,vv in new_sorted_obb if kk==uk])/5) for uk in set([k for k,v in new_sorted_obb])]
	print("new collected obb after summing probabilities and averaging")
	new_collected_obb = sorted(new_collected_obb)
	print(new_collected_obb)
	print(len(new_collected_obb))
	print('\n')

	for word in new_collected_obb:
		for element in filtered_sentence:
			if word[0] == element:
				if word[0] not in present:
					present.append(word[0])

	for word in new_fasttext_return:
		for element in filtered_sentence:
			if word[0] == element:
				if word[0] not in present:
					present.append(word[0])			



	ft_len = len(new_fasttext_return)
	w2v_len = len(new_collected_obb)
	i = 0

	while i < w2v_len:
		j = 0
		while j < ft_len:
			if new_fasttext_return[j][1] > 0.70:
				if new_fasttext_return[j][0] not in maybe_present:
						maybe_present.append(new_fasttext_return[j][0])
			sim = (model.wv.similarity(new_collected_obb[i][0], new_fasttext_return[j][0]))
			if sim > 0.4 :
				if new_collected_obb[i][0] not in maybe_present:
					maybe_present.append(new_collected_obb[i][0]) 
			#+ ", " + new_collected_obb[i][0] + ", " + new_fasttext_return[j][0]
			j = j + 1
		i = i + 1

	print("words definitely present are:")
	print(present)
	print('\n')

	print("words maybe present are:")
	print(maybe_present)
	print('\n')

	cluster = present + maybe_present
	# for i in range (0,20):
	# 	new_clust.append(cluster[i])
	new_clust = ' '.join(cluster)
	# # print(clsuter)

	# p = Popen(['python3 ann2.py'], stdin=PIPE, shell=True)
	# p.communicate(new_clust.encode())

	cluster_class = classify(new_clust)
	print("cluster class is ", cluster_class)	


print ("enter:")
testvar = input()
preprocessing(testvar)
strr = " ".join(filtered_sentence)
backend(strr)	








#print(strr)

# print(filtered_sentence)





# r = Popen(['cd fastText','python3 shell.py'], stdin=PIPE, shell=True)
# r.communicate(strr.encode())



# shell_return = shell.returns_lists(strr)




# n = len(shell_return)
# i = 0
# while i < n:
# 	m = len(shell_return[i])
# 	j = 0
# 	while j < m:
# 		new_shell_return.append(shell_return[i][j])
# 		j = j + 1
# 	i = i + 1




# print("shell returned list:")
# print (new_shell_return)
# print(len(new_shell_return))


# q = Popen(['python fast.py'], stdin=PIPE, shell=True)
# q.communicate(strr.encode())

 

#the film was emotional and had a deep effect





# print (obb)


#shell_return = returns(strr)
#print(shell_return)



#replace directory with your desired directory
# for i in Main_Path.walk():
#     if i.isfile():
#         if i.name == 'parrot.pk':
#             i.remove()






# print(new_obb)
# print(len(new_obb))

# print(new_obb)

# pp = 0
# for i, (a, b) in enumerate(new_obb):
#     if a not in new_sorted_obb:
#     	new_sorted_obb.append((a,b))
#     else:
#     	if (a, _) in new_sorted_obb:
#     	   pp = new_sorted_obb.index(i)	
#     new_sorted_obb[pp][1] = new_sorted_obb[pp][1] + b

#     #tuple_list[i] = (a, new_b)
# print("new sorted obb:")

# print(new_sorted_obb)
# print(len(new_sorted_obb))










































# print("new collective obb after averaging:	")
# # print(new_collected_obb)
# divisor = 3.0
# new_averaged_obb = tuple(((x/3.0) for x in new_collected_obb[i][1]) for i in range(0, len(new_collected_obb)))

# # new_averaged_obb = sorted(new_averaged_obb,  key=lambda x: x[1], reverse = True)

# # print(new_averaged_obb)
# # print(len(new_averaged_obb))
# # print('\n')



# # y = " ".join(filtered_sentence)
# # y.encode('utf-8')
# # print(y)
# # 


# # returned_words.append(model.predict_output_word(filtered_sentence))
# # j = 0
# # while j <10 :
# # 	returned = model.predict_output_word(filtered_sentence)
# # 	i = 0
# # troop = 0




# # k = len(obb)
# # print(k)

# # for i in range(0, k - 1):
# # 	if obb[i][0] not in returned_words:
# # 		print(obb[i])
# # 		for j in obb[i][0]:
# # 			print(j)s




# 			# returned_words.append(obb[i][0])
# 			# returned_prob.append(obb[i][1])
# 	# else:
# 	# 	for j in range (0, len(returned_words)):
# 	# 		if obb[i][0] == returned_words[j][0]:
# 	# 			returned_prob[j][0] = returned_prob[j][0] + obb[i][1]






# # for i in range (0, k-1):
# # 	print(i)
# # 	if obb[i][0] not in returned_words:
# # 		print(obb[i])
# # 		returned_words.append(obb[i][0])
# # 		returned_prob.append(obb[i][1])
# # 	else:
# # 		for j in range (0, len(returned_words)):
# # 			if obb[i][0] == returned_words[j][0]:
# # 				returned_prob[j][0] = returned_prob[j][0] + obb[i][1]

# # for i in range (0, len(returned_prob)):
# # 	returned_prob[i] = returned_prob[i] / 10

# # Z = [x for _,x in sorted(zip(returned_prob,returned_words), reverse=True)]
# # print(Z)
# # print(returned_prob.sorted(reverse=True))





# 	# j = j+1	
# 	# print (j)	

# # print(returned_words)

# # print(returned_words)

# # print(k)

# # print(returned_prob)
# #print(model.predict_output_word(filtered_sentence))
