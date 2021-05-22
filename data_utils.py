import random


class CRMatchingDataset:

    def __init__(self, contexts, responses, labels=None, batch_size=20, shuffle=False):
    
        self.contexts = contexts
        self.responses = responses
        self.labels = labels
        self.batch_size = batch_size
        self.index = 0
        
        assert len(contexts) == len(responses) == len(labels)
        #assert len(labels) % self.batch_size == 0
        
        if shuffle:
            tmp = list(zip(self.contexts, self.responses, self.labels))
            random.shuffle(tmp)
            self.contexts[:], self.responses[:], self.labels[:] = zip(*tmp)

    def next(self):
        contexts = self.contexts[self.index:self.index + self.batch_size]
        responses = self.responses[self.index:self.index + self.batch_size]
        labels = self.labels[self.index:self.index + self.batch_size]
        if self.index + self.batch_size >= len(self.labels):
            self.index = 0
        else:
            self.index += self.batch_size
        return contexts, responses, labels
        
    def __len__(self):
        return len(self.labels)
    
    def batches(self):
        return int((len(self.labels) + self.batch_size - 1) / self.batch_size)

        
if __name__ == '__main__':
    path = '../../data/msn-version/ubuntu_data/'
    
    # import pickle as pkl
    # train_contexts, train_responses, train_labels = pkl.load(file=open(path + "train.pkl", 'rb'))
    # dev_contexts, dev_responses, dev_labels = pkl.load(file=open(path + "dev.pkl", 'rb'))
    # vocab, word_embeddings = pkl.load(file=open(path + "vocab_and_embeddings.pkl", 'rb'))
    
    # breakpoint() # glance data structure
    