import numpy as np
import re
import itertools
from collections import Counter
import glob
import collections



def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(createFile,dataSet):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    posFiles = "./aclImdb/"+dataSet+"/pos/*.txt"
    posOutput= "./aclImdb/"+dataSet+"/pos/result.txt"
    negFiles = "./aclImdb/"+dataSet+"/neg/*.txt"
    negOutput= "./aclImdb/"+dataSet+"/neg/result.txt"
    print("heck path:") 
    print(posOutput)
    print(negOutput) 

    # Load data from files 
    if(createFile): 
        read_files = glob.glob(posFiles)
        with open(posOutput, "wb") as outfile:
            for f in read_files:
                with open(f, "rb") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")
    positive_examples = list(open(posOutput, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]

    if(createFile): 
        read_files = glob.glob(negFiles)
        with open(negOutput, "wb") as outfile:
            for f in read_files:
                with open(f, "rb") as infile:
                    outfile.write(infile.read())
                outfile.write("\n")
    negative_examples = list(open(negOutput, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    #vocabulary = create_vocabulary(" ".join(x_text).split())
    #process_text = substitute_oov(x_text,vocabulary)
    return [x_text, y]


def create_vocabulary(words):
    vocabulary_size=10000
    count = [('oov', -1)]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    vocabulary = set()
    for element in count:
        vocabulary.add(element[0]) 
    return vocabulary

def substitute_oov(text,vocabulary):
    process_text = []
    #print "befor loops"
    #i=0
    for sample in text:
        #i = i+1
        processed_sample = []
        for word in sample.split(" "):
            if word in vocabulary:
                processed_sample.append(word)
            else:
                processed_sample.append("oov")
        processed_sample = " ".join(processed_sample)        
        process_text.append(processed_sample)
        #if i%100==0:
        #    print i
    #print "after loops"
    return process_text

#def replace_all(text, vocabulary):
#    process_text = []
#    for t in text:
#        print t
#        for i in vocabulary:
#            t = t.replace(i,'oov')
#        print t
#        exit()
#        process_text.append(t)
#    return process_text



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]




