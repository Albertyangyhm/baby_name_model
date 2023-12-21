import numpy as np

class LDAGibbsSampler:
    def __init__(self, num_topics, docs, alpha, eta, gamma, val_docs=None):
        self.K = num_topics      
        self.docs = docs            

        self.alpha = alpha          
        self.eta = eta              
        self.gamma = gamma

        self.V = len(docs[0])       
        self.D = len(docs)
        self.val_docs = val_docs

        self.z = [[np.random.choice(self.K) for _ in range(len(doc))] for doc in docs]

        
        self.topic_count = np.zeros(self.K)
        self.topic_document_count = np.zeros((self.K, self.D))

        self.topic_word_count = np.zeros((self.K, self.V))
        
        for d, doc in enumerate(docs):
            for n, word_count in enumerate(doc):
                for _ in range(word_count):
                    topic = self.z[d][n]
                    self.topic_document_count[topic, d] += 1
                    self.topic_word_count[topic, n] += 1
                    self.topic_count[topic] += 1
        
        self.best_model_state = None
        self.best_metric = float('inf')
        self.best_epoch = 0

    def _get_current_state(self):
        return {
            'topic_word_count': np.copy(self.topic_word_count),
            'topic_count': np.copy(self.topic_count),
            'topic_document_count': np.copy(self.topic_document_count),
            'z': [list(doc) for doc in self.z]
        }

    def compute_log_likelihood(self):
        log_likelihood = 0
        for d in range(self.D):
            for n in range(self.V):
                word_count = self.docs[d][n]
                if word_count > 0:
                    log_likelihood += word_count * np.log(np.sum(
                        ((self.topic_count + self.alpha) / (np.sum(self.topic_count) + self.K * self.alpha)) * \
                        ((self.topic_document_count[:, d] + self.gamma) / (self.topic_count + self.D * self.gamma)) * \
                        ((self.topic_word_count[:, n] + self.eta) / (self.topic_count + self.V * self.eta))
                    ))
        return log_likelihood

    def compute_perplexity(self, docs):
        log_likelihood = 0
        word_count = 0
        for d in range(len(docs)):
            for n in range(self.V):
                count = docs[d][n]
                if count > 0:
                    word_count += count
                    log_likelihood += count * np.log(np.sum(
                        ((self.topic_count + self.alpha) / (np.sum(self.topic_count) + self.K * self.alpha)) * \
                        ((self.topic_document_count[:, d] + self.gamma) / (self.topic_count + self.D * self.gamma)) * \
                        ((self.topic_word_count[:, n] + self.eta) / (self.topic_count + self.V * self.eta))
                    ))
        perplexity = np.exp(-log_likelihood / word_count)
        return perplexity

    def sample(self, num_iterations):
        train_llhs = []
        for i in range(num_iterations):
            for d in range(self.D):
                for n in range(self.V):
                    word_count = self.docs[d][n]
                    if word_count > 0:
                        # Remove current word's topic assignment
                        old_topic = self.z[d][n]

                        self.topic_word_count[old_topic, n] -= word_count
                        self.topic_count[old_topic] -= word_count
                        self.topic_document_count[old_topic, d] -= word_count

                        # Compute conditional distribution for topics
                        p_z = (
                            ((self.topic_count + self.alpha) / (np.sum(self.topic_count) + self.K * self.alpha)) * \
                            ((self.topic_document_count[:, d] + self.gamma) / (self.topic_count + self.D * self.gamma)) * \
                            ((self.topic_word_count[:, n] + self.eta) / (self.topic_count + self.V * self.eta))
                        )
                        # Normalize
                        p_z /= np.sum(p_z)

                        # Sample new topic and update counts
                        new_topic = np.random.choice(self.K, p=p_z)
                        self.z[d][n] = new_topic
                        self.topic_word_count[new_topic, n] += word_count
                        self.topic_count[new_topic] += word_count
                        self.topic_document_count[new_topic, d] += word_count 
        
            train_log_likelihood = self.compute_log_likelihood()
            print(f"Epoch {i+1}, Training Log-Likelihood: {train_log_likelihood}")
            train_llhs.append(train_log_likelihood)

            if train_log_likelihood < self.best_metric:
                self.best_metric = train_log_likelihood
                self.best_model_state = self._get_current_state()
                self.best_epoch = i + 1

            # if self.val_docs is not None:
            #     # Compute and print perplexity for validation data
            #     val_perplexity = self.compute_perplexity(self.val_docs)
            #     print(f"Epoch {i+1}, Validation Perplexity: {val_perplexity}")

        return train_llhs


