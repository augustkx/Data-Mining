# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:31:41 2016

@author: Jeroen
"""

#LambdaMART

#%% LambdaMART model
#use nDCG and 5-fold crossvalidation for eval

NUM_EPOCHS = 500
BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.00005
MOMENTUM = 0.95
NUM_FEATURES = len(features)


def lambda_loss(output, lambdas):
    return theano.tensor.mul(output, lambdas)

# calculate the maxNDCG for listwise algorithm 
def maxNDCG(labels):
    labels.sort()
    lbls = np.asarray(labels[::-1])
            
    # sort them and calculate the maximum ndcg
    maxNDCG = nDCGAtTen(lbls)
    return maxNDCG

class LambdaRankHW:
    NUM_INSTANCES = count()

    def __init__(self, feature_count, algorithm):
        
        # on initialization functions
        self.algorithm = algorithm
        self.feature_count = feature_count
        self.output_layer = self.build_model(feature_count,1,BATCH_SIZE)
        self.iter_funcs = self.create_functions(self.output_layer)
        

    # train_queries are what load_queries returns - implemented in query.py
    def train_with_queries(self, train_queries, num_epochs):
        try:
            now = time.time()
            for epoch in self.train(train_queries):
                if epoch['number'] % 1 == 0:
                    print("Epoch {} of {} took {:.3f}s".format(
                    epoch['number'], num_epochs, time.time() - now))
                    now = time.time()
                    print("training loss:\t\t{:.6f}\n".format(epoch['train_loss']))
                    
                if epoch['number'] >= num_epochs:
                    break
        except KeyboardInterrupt:
            pass

    def score(self, query):
        feature_vectors = query.get_feature_vectors()
        scores = self.iter_funcs['out'](feature_vectors)
        return scores


    def build_model(self,input_dim, output_dim,
                    batch_size=BATCH_SIZE):
        """Create a symbolic representation of a neural network with `intput_dim`
        input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
        layer.

        The training function of this model must have a mini-batch size of
        `batch_size`.

        A theano expression which represents such a network is returned.
        """
        print("input_dim",input_dim, "output_dim",output_dim)
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, input_dim),
        )

        l_hidden = lasagne.layers.DenseLayer(
            l_in,
            num_units=200,
            nonlinearity=lasagne.nonlinearities.tanh,
        )


        l_out = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.linear,
        )

        return l_out

    # Create functions to be used by Theano for scoring and training
    def create_functions(self, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM, L1_reg=0.0000005, L2_reg=0.000003):
        """Create functions for training, validation and testing to iterate one
           epoch.
        """
        X_batch = X_tensor_type('x')
        y_batch = T.fvector('y')

        output_row = lasagne.layers.get_output(output_layer, X_batch, dtype="float32")
        output = output_row.T

        output_row_det = lasagne.layers.get_output(output_layer, X_batch,deterministic=True, dtype="float32")
        print(self.algorithm)
        
        
        # Point-wise loss function (squared error)
        if self.algorithm == 'pointwise':
            loss_train = lasagne.objectives.squared_error(output,y_batch)
        # Pairwise loss function - comment it in
        else:
            loss_train = lambda_loss(output,y_batch)

        loss_train = loss_train.mean()

        #add regularization
        L1_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l1)
        L2_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l2)
        loss_train = loss_train.mean() + L1_loss * L1_reg + L2_loss * L2_reg

        # Parameters you want to update
        all_params = lasagne.layers.get_all_params(output_layer)

        # Update parameters, adam is a particular "flavor" of Gradient Descent
        updates = lasagne.updates.adam(loss_train, all_params)


        # Create two functions:

        # (1) Scoring function, deterministic, does not update parameters, outputs scores
        score_func = theano.function(
            [X_batch],output_row_det,
        )

        # (2) Training function, updates the parameters, outpust loss
        train_func = theano.function(
            [X_batch,y_batch], loss_train,
            updates=updates,
            #givens={
                 #X_batch: dataset['X_train'][batch_slice],
                 # y_batch: dataset['y_valid'][batch_slice],
             #},
        )

        print("finished create_iter_functions")
        return dict(
            train=train_func,
            out=score_func,
        )


def lambda_function_listwise(self,labels, scores):
        '''
        Function that implements the listwise algorithm
        '''
        
        
        def sigmoid(u, v, S):
            return (-1/(1+np.exp(S*(u- v))))
        
        def compute_Suv(u,v):
            # compare values and outpu a vector of -1,0,1
            different = np.where(u == v, 0, 1)
            smaller = np.where(u < v, -1, 1)
            
            return different*smaller, different
        
        # calculate the  absolute difference in NDCG
        def delta_NDCG(NDCG_now, swapped_list, max_ndcg):
            return abs(NDCG_now - nDCGAtTen(swapped_list, max_ndcg))
            
        # swap the list
        def swap(u,v,lbs):
            temp = list(lbs)
            temp[u], temp[v] = temp[v], temp[u]
            return temp

        lambdas_per_doc = []
        
        # store the maxNDCG
        maxlbls = maxNDCG(labels)
        NDCG = nDCGAtTen(labels,maxlbls)
        
        # if the list is empty we needn't bother with it!
        if NDCG == 0:
            return np.asarray(np.zeros(len(scores)), dtype= "float32")
        
        # iterate over all the documents in the query
        for doc_v in range(len(scores)):
            
            # compute S
            S_uv, smallerLarger = compute_Suv(labels[doc_v], labels) 
            
            # determine sigmoid output
            sigm_output = sigmoid(scores[doc_v], scores, S_uv)
            
            delta_ndcg = []
            
            # now calculate each delta NDCG score
            for x in range(len(scores)):
                if labels[doc_v] != labels[x]:
                    swappedList = swap(doc_v, x, labels)
                    delta = delta_NDCG(NDCG, swappedList, maxlbls)
                    delta_ndcg.append(delta)
                else:
                    delta_ndcg.append(0)
                    
            # Multiply delta NDCG by the sigmoid functions outpt
            lambda_u_v = sigm_output* np.asarray(delta_ndcg)
            temp = np.dot(lambda_u_v, smallerLarger)


            # finally calculate lambdas per document
            lambdas_per_doc.append(temp)
            
        return np.asarray(lambdas_per_doc, dtype= "float32")
        


    def compute_lambdas_theano(self,query, labels):
        scores = self.score(query).flatten()
        if self.algorithm =='pairwise':
            result = self.lambda_function_pairwise(labels, scores[:len(labels)])
        else:
            result = self.lambda_function_listwise(labels, scores[:len(labels)])
            
        return result

    def train_once(self, X_train, query, labels):

        # Calculate Lambdas
        if self.algorithm == 'pairwise' or self.algorithm == 'listwise':
            lambdas = self.compute_lambdas_theano(query,labels)
            lambdas.resize((BATCH_SIZE, ))

        X_train.resize((BATCH_SIZE, self.feature_count),refcheck=False)

        #to replace labels by lambdas
        if self.algorithm == 'pairwise' or self.algorithm == 'listwise' :
            batch_train_loss = self.iter_funcs['train'](X_train, lambdas)
        elif self.algorithm == 'pointwise':
            batch_train_loss = self.iter_funcs['train'](X_train, labels)
        return batch_train_loss


    def train(self, train_queries):
        X_trains = train_queries.get_feature_vectors()

        queries = train_queries.values()

        for epoch in itertools.count(1):
            batch_train_losses = []
            random_batch = np.arange(len(queries))
            np.random.shuffle(random_batch)
            for index in xrange(len(queries)):
                random_index = random_batch[index]
                labels = queries[random_index].get_labels()
                
                # skips over iterations if batch size is not large enough
                # Checks which algorithm to use 
                X_train = X_trains[random_index]
                if len(X_train) == BATCH_SIZE and self.algorithm == 'pairwise':
                    
                     # use pairwise algorithm
                     batch_train_loss = self.train_once(X_trains[random_index],queries[random_index],labels)
                     batch_train_losses.append(batch_train_loss)
                     
                elif len(X_train) == BATCH_SIZE and self.algorithm == 'pointwise':
                    
                    # use pointwise algorithm
                    batch_train_loss = self.train_once(X_train,queries[random_index],labels)
                    batch_train_losses.append(batch_train_loss)
                    
                elif len(X_train) == BATCH_SIZE and self.algorithm == 'listwise':
                    
                    # use listwise algorithm
                     batch_train_loss = self.train_once(X_trains[random_index],queries[random_index],labels)
                     batch_train_losses.append(batch_train_loss)
                else:
                    continue

            avg_train_loss = np.mean(batch_train_losses)

            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
            }



def nDCGAtTen(Ranking, max_score  = 1):
    '''
    Function which calculates the nDCG at 10 for a vector of judgements
    Returns Normalized nDCG score
    
    '''
    
    # global value for the one you need to subtract in the DCG formula
    allOnes = np.ones((1,len(Ranking)))
    
    # Maximum possible score
    Max_score = max_score
    
    # calculate the discount rate once
    discount_rate = np.log2(np.array(range(0, len(Ranking) + 2)[2:], dtype= float))
    
    # Function to calculate the difference in DCG scores
    def calculateDifferenceDCG(Ranking):
        
        # Formula: 2^relevance - 1 
        gain_E = np.subtract(np.power(2, Ranking), allOnes)
        
        # Complete the discounted gains
        disc_gain_E =  np.sum(gain_E/ discount_rate[:])
        
        
        return  disc_gain_E /Max_score
    
    # Iterate over list with numpy arrays
    normalizedDCG =calculateDifferenceDCG(Ranking)
    
    return normalizedDCG

    

def cross_validate(iterations, train, test, valid, algorithm):
    '''
    Script that will run cross validation over different validation sets
    '''
    
    trainingQueries = query.load_queries(train, NUM_FEATURES)
    testQueries = query.load_queries(test, NUM_FEATURES)
    validationQueries = query.load_queries(valid, NUM_FEATURES)
    
    # get the feature vectors and values
    validationLabels = validationQueries.values()
    testLabels = testQueries.values()
    
    # initialize the neural network
    Ranker = LambdaRankHW(NUM_FEATURES, algorithm)
    
    # keep track of scores for test and validation set.
    average_score_valid = 0
    average_score_test = 0 
    
    # train the neural network and cross validate
    for x in range(iterations):
        Ranker.train_with_queries(trainingQueries, 1)
        
        # lists of scores
        scoreList_valid = []
        scoreList_test = []
        
        length_validation = len(validationLabels)
        length_test = len(testLabels)
        
        # get the scores of query
        validationValues = validationQueries.values()
        testValues = testQueries.values()
        
        
        # Run over all queries in the validation set
        for index in range(len(validationLabels)):
            
            # calculate the max DCG
            # get the labels
            labels = validationLabels[index].get_labels().tolist()
            labels.sort()
            labels = np.asarray(labels[::-1])
            
            # sort them and calculate the maximum ndcg
            maxNDCG = nDCGAtTen(labels[:10])
            
            #print "NDCG: ", maxNDCG 
            # if no score is possible in a query skip to next
            if maxNDCG == 0:
                length_validation -= 1
                continue
            
            lbls = validationLabels[index].get_labels().tolist()
            score = Ranker.score(validationValues[index])
            
            
            list_of_tuples = zip(lbls, score)
            list_of_tuples.sort(key=lambda tup: tup[1], reverse=True)
            
            score = [list(t) for t in zip(*list_of_tuples)][0]
            
            NDCG = nDCGAtTen(score[:10], maxNDCG)
            # now score the documents using network and ndcg
            scoreList_valid.append(NDCG)
            
            
            
        # Run over all queries in the test set
        for index in range(len(testLabels)):
            
            # calculate the max DCG
            # get the labels
            labels = testLabels[index].get_labels().tolist()
            labels.sort()
            labels = np.asarray(labels[::-1])
            
            # sort them and calculate the maximum ndcg
            maxNDCG = nDCGAtTen(labels[:10])
            
            #print "NDCG: ", maxNDCG 
            # if no score is possible in a query skip to next
            if maxNDCG == 0:
                length_test -= 1
                continue
            
            lbls = testLabels[index].get_labels().tolist()
            score = Ranker.score(testValues[index])
            
            
            list_of_tuples = zip(lbls, score)
            list_of_tuples.sort(key=lambda tup: tup[1], reverse=True)
            
            score = [list(t) for t in zip(*list_of_tuples)][0]
            
            NDCG = nDCGAtTen(score[:10], maxNDCG)
            # now score the documents using network and ndcg
            scoreList_test.append(NDCG)
        
        # calculate the averages of the scores after each iteration
        average_score_valid = sum(scoreList_valid) / length_validation
        average_score_test = sum(scoreList_test) / length_test
        print "Score validation set: ", average_score_valid
        print "Score test set: ", average_score_test
        
    return average_score_valid, average_score_test
            
            
def plot(test, validation):
    # Create a barchart

    n_groups = 5
    bar_width = 0.3
    opacity = 0.4
    index = np.arange(n_groups)
    fig, ax = plt.subplots()
    test = plt.bar(index, test, bar_width,
                     alpha=opacity,
                     color='b',
                     label='test data')
    validation = plt.bar(index + bar_width, validation, bar_width,
                     alpha=opacity,
                     color='r',
                     label='validation data')
    
    
    plt.xlabel('Folder')
    plt.ylabel('NDCG score ')
    plt.title('NDCG scores cross vallidaiton')
    plt.xticks(index + bar_width, ('0','1', '2', '3', '4', '5','6','7','8','9'))
    plt.legend()
    plt.legend(bbox_to_anchor=(1.3, 1.05))

    plt.tight_layout()
    plt.show()
    
    
    

def main():
    # 5 point cross validation for the pairwise algorithm
    test_point = []
    valid_point = []
    for x in range(len(TRAIN_FILE)):
        test_score, validation_score = cross_validate(10,TRAIN_FILE[x], TEST_FILE[x], VALID_FILE[x], 'pointwise')
        test_point.append(test_score)
        valid_point.append(validation_score)
    
    
    # 5 point cross validation for the pointwise algorithm
    test_pair = []
    valid_pair = []
    for x in range(len(TRAIN_FILE)):
        test_score, validation_score = cross_validate(10,TRAIN_FILE[x], TEST_FILE[x], VALID_FILE[x], 'pairwise')
        test_pair.append(test_score)
        valid_pair.append(validation_score)
#    
    
    # 5 point cross validation for the pointwise algorithm
    test_list = []
    valid_list = []
    for x in range(len(TRAIN_FILE)):
        test_score, validation_score = cross_validate(10,TRAIN_FILE[x], TEST_FILE[x], VALID_FILE[x], 'listwise')
        test_list.append(test_score)
        valid_list.append(validation_score)
    
#    # now plot it!
    plot(test_point, valid_point)
    plot(test_pair, valid_pair)
    plot(test_list, valid_list)
   
    print "Pointwise average Validation score:", sum(valid_point)/float(len(valid_point))
    print "Pointwise average Test score:", sum(test_point)/float(len(test_point))
    print "Pairwise average Validation score:", sum(valid_pair)/float(len(valid_pair))
    print "Pairwise average Test score:", sum(test_pair)/float(len(test_pair))
    print "Listwise average Validation score:", sum(valid_list)/float(len(valid_list))
    print "Listwise average Test score:", sum(test_list)/float(len(test_list))