import tensorflow as tf
import numpy as np
class Generator(object):
    """SeqGAN implementation based on https://arxiv.org/abs/1609.05473
        "SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient"
        Lantao Yu, Weinan Zhang, Jun Wang, Yong Yu
    """
    def __init__(self, config):
        """ Basic Set up

        Args:
           num_emb: output vocabulary size
           batch_size: batch size for generator
           emb_dim: LSTM hidden unit dimension
           sequence_length: maximum length of input sequence
           start_token: special token used to represent start of sentence
           initializer: initializer for LSTM kernel and output matrix
        """
        self.num_emb = config.num_emb
        self.batch_size = config.gen_batch_size
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.sequence_length = config.sequence_length
        self.start_token = tf.constant(config.start_token, dtype=tf.int32, shape=[self.batch_size])
        self.initializer = tf.random_normal_initializer(stddev=0.1)


    def build_input(self, name):
        """ Buid input placeholder

        Input:
            name: name of network
        Output:
            self.input_seqs_pre (if name == pretrained)
            self.input_seqs_mask (if name == pretrained, optional mask for masking invalid token)
            self.input_seqs_adv (if name == 'adversarial')
            self.rewards (if name == 'adversarial')
        """
        assert name in ['pretrain', 'adversarial', 'sample', 'arm']
        if name == 'pretrain':
            self.input_seqs_pre = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_seqs_pre")
            self.input_seqs_mask = tf.placeholder(tf.float32, [None, self.sequence_length], name="input_seqs_mask")
        elif name == 'adversarial':
            self.input_seqs_adv = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_seqs_adv")
            self.rewards = tf.placeholder(tf.float32, [None, self.sequence_length], name="reward")
        elif name == 'arm':
            self.pi_sample_input = tf.placeholder(tf.float32, [self.sequence_length, self.batch_size, self.num_emb], name="pi_sample_input")
            self.rewards_list = tf.placeholder(tf.float32, [self.sequence_length, self.batch_size, self.num_emb], name="rewards_list")

    def build_pretrain_network(self):
        """ Buid pretrained network

        Input:
            self.input_seqs_pre
            self.input_seqs_mask
        Output:
            self.pretrained_loss
            self.pretrained_loss_sum (optional)
        """
        self.build_input(name="pretrain")
        self.pretrained_loss = 0.0
        with tf.variable_scope("teller"):
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", [self.num_emb, self.emb_dim], "float32", self.initializer)
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [self.hidden_dim, self.num_emb], "float32", self.initializer)

            with tf.variable_scope("lstm"):
                for j in range(self.sequence_length):
                    with tf.device("/cpu:0"):
                        if j == 0:
                            # <BOS>
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.start_token)
                        else:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.input_seqs_pre[:, j-1])
                    if j == 0:
                        state = lstm1.zero_state(self.batch_size, tf.float32)

                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())

                    logits = tf.matmul(output, output_W)
                    # calculate loss
                    pretrained_loss_t = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_seqs_pre[:,j], logits=logits)
                    pretrained_loss_t = tf.reduce_sum(tf.multiply(pretrained_loss_t, self.input_seqs_mask[:,j]))
                    self.pretrained_loss += pretrained_loss_t
                    word_predict = tf.to_int32(tf.argmax(logits, 1))
            self.pretrained_loss /= tf.reduce_sum(self.input_seqs_mask)
            self.pretrained_loss_sum = tf.summary.scalar("pretrained_loss",self.pretrained_loss)

    def build_adversarial_network(self):
        """ Buid adversarial training network

        Input:
            self.input_seqs_adv
            self.rewards
        Output:
            self.gen_loss_adv
        """
        self.build_input(name="adversarial")
        self.softmax_list_reshape = []
        self.softmax_list = []
        with tf.variable_scope("teller"):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", [self.num_emb, self.emb_dim], "float32", self.initializer)
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [self.hidden_dim, self.num_emb], "float32", self.initializer)
            with tf.variable_scope("lstm"):
                for j in range(self.sequence_length):
                    tf.get_variable_scope().reuse_variables()
                    with tf.device("/cpu:0"):
                        if j == 0:
                            # <BOS>
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.start_token)
                        else:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.input_seqs_adv[:, j-1])
                    if j == 0:
                        state = lstm1.zero_state(self.batch_size, tf.float32)
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())

                    logits = tf.matmul(output, output_W)
                    softmax = tf.nn.softmax(logits)
                    self.softmax_list.append(softmax)
            self.softmax_list_reshape = tf.transpose(self.softmax_list, perm=[1, 0, 2])
            self.gen_loss_adv = -tf.reduce_sum(
                tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tf.reshape(self.input_seqs_adv, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                        tf.clip_by_value(tf.reshape(self.softmax_list_reshape, [-1, self.num_emb]), 1e-20, 1.0)
                    ), 1) * tf.reshape(self.rewards, [-1]))

    def build_arm_gradient(self):
        """ Compute arm gradient"""
        self.build_input(name="arm")
        self.softmax_list_reshape = []
        self.arm_gradients = []
        self.softmax_list = []
        num_emb = self.num_emb
        # first generate samples and remember the pi samples
        with tf.variable_scope("teller"):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", [self.num_emb, self.emb_dim], "float32", self.initializer)
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [self.hidden_dim, self.num_emb], "float32", self.initializer)
            with tf.variable_scope("lstm"):
                for j in range(self.sequence_length):
                    arm_gradients_tmp = tf.multiply(tf.transpose(tf.transpose(self.rewards_list[j,:,:]) - tf.reduce_mean(self.rewards_list[j,:,:], axis=1)), 1 - self.num_emb * self.pi_sample_list[j,:, num_emb-1:num_emb])
                    arm_gradients_tmp = - tf.concat([arm_gradients_tmp[:, 0:num_emb - 1], np.zeros([self.batch_size, 1])], 1)

                    # #candidate_reward.append([self.rewards_list[j - 1, entry, num_emb] for entry in tf.argmin(pi_sample * exp_logits, axis=1)])
                    # candidate_move = tf.argmin(pi_sample * exp_logits, axis=1)
                    # indices = tf.reshape(tf.concat([np.tile(j - 1, self.batch_size), np.arange(self.batch_size), candidate_move], axis=0), [-1,self.batch_size])
                    # indices = tf.transpose(indices)
                    # candidate_reward_tmp = tf.gather_nd(self.rewards_list, indices, name=None)
                    # candidate_reward.append(candidate_reward_tmp)
                    # candidate_reward = tf.transpose(tf.convert_to_tensor(candidate_reward))
                    # arm_gradients_tmp = tf.multiply((candidate_reward - tf.reduce_mean(candidate_reward, axis=0)), (1 - num_emb * pi_sample[:, num_emb - 1: num_emb]))
                    # arm_gradients_tmp = tf.concat([arm_gradients_tmp[:, 0:num_emb - 1], np.zeros([self.batch_size, 1])], 1)
                    self.arm_gradients.append(arm_gradients_tmp)
                    # sequence_length - 1 * batch_size * num_emb
            self.arm_gradients = tf.convert_to_tensor(self.arm_gradients, dtype=tf.float32)


    def build_sample_network(self):
        """ Buid sampling network

        Output:
            self.sample_word_list_reshape
        """
        self.build_input(name="sample")
        self.sample_word_list = []
        self.sample_arm_word_list = []
        self.pi_sample_list = []
        self.logits_list = []
        with tf.variable_scope("teller"):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", [self.num_emb, self.emb_dim], "float32", self.initializer)
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [self.hidden_dim, self.num_emb], "float32", self.initializer)

            with tf.variable_scope("lstm"):
                pi_sample_list = []
                for j in range(self.sequence_length):
                    pi_sample_list.append(np.float32(np.random.dirichlet(np.ones(self.num_emb), size=self.batch_size)))
                for j in range(self.sequence_length):
                    with tf.device("/cpu:0"):
                        if j == 0:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.start_token)
                        else:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, sample_word)
                    if j == 0:
                        input_state = lstm1.zero_state(self.batch_size, tf.float32)
                    output, output_state = lstm1(lstm1_in, input_state, scope=tf.get_variable_scope())
                    logits = tf.matmul(output, output_W)
                    exp_logits = tf.exp(-logits)
                    pi_sample = pi_sample_list[j]
                    sample_word = tf.argmin(pi_sample * exp_logits, axis=1)
                    sample_arm_word_per_sequence = []
                    self.logits_list.append(logits)
                    for k in range(self.num_emb):
                        pi_sample_swap = np.copy(pi_sample)
                        pi_sample_swap[:, k] = pi_sample[:, self.num_emb - 1]
                        pi_sample_swap[:, self.num_emb - 1] = pi_sample[:, k]
                        candidate_move = tf.argmin(pi_sample_swap * exp_logits, axis=1)
                        sample_arm_word_per_embedding = self.sample_word_list.copy()
                        sample_arm_word_per_embedding.append(candidate_move)
                        for l in range(j, self.sequence_length):
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, candidate_move)
                            if l == j:
                                state_copy = input_state
                            output, state_copy = lstm1(lstm1_in, state_copy, scope=tf.get_variable_scope())
                            logits = tf.matmul(output, output_W)
                            exp_logits = tf.exp(-logits)
                            pi_sample = pi_sample_list[l]
                            sample_arm_word_per_embedding.append(tf.argmin(pi_sample * exp_logits, axis=1))   # sequence_length * batch_size
                        sample_arm_word_per_sequence.append(sample_arm_word_per_embedding) #    num_emb * sequene_length * batch_size
                    input_state = output_state
                    #logprob = tf.log(tf.nn.softmax(logits))
                    #sample_word = tf.reshape(tf.to_int32(tf.multinomial(logprob, 1)), shape=[self.batch_size])
                    self.pi_sample_list.append(pi_sample)
                    self.sample_word_list.append(sample_word) #sequence_length * batch_size
                    self.sample_arm_word_list.append(sample_arm_word_per_sequence) # sequence_length * num_emb * sequence_length * batch_size
            self.logits_list = tf.convert_to_tensor(self.logits_list, dtype=tf.float32)
            self.sample_word_list_reshape = tf.transpose(tf.squeeze(tf.stack(self.sample_word_list)), perm=[1,0])
            self.sample_arm_word_list_reshape = tf.transpose(tf.squeeze(tf.stack(self.sample_arm_word_list)), perm=[0, 3, 1, 2]) #batch_size * sequene_length * num_emb * sequence_length
            self.pi_sample_list = tf.squeeze(tf.stack(self.pi_sample_list))
    def build(self):
        """Create all network for pretraining, adversairal training and sampling"""
        self.build_pretrain_network()
        self.build_adversarial_network()
        self.build_sample_network()
        self.build_arm_gradient()
    def generate(self, sess):
        """Helper function for sample generation"""
        return sess.run(self.sample_word_list_reshape)
