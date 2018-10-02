import random
import os
import pickle as cPickle
import numpy as np
import tensorflow as tf
import collections
import time
from configuration import *
from utils import *
from dataloader import Gen_Data_loader, Dis_dataloader
from discriminator import Discriminator
from generator import Generator
from rollout import rollout
from target_lstm import TARGET_LSTM
import bleu
#Hardware related setting


config_hardware = tf.ConfigProto()
config_hardware.gpu_options.per_process_gpu_memory_fraction = 0.40
os.environ["CUDA_VISIBLE_DEVICES"]="0"



def main(unused_argv):
    config_train = training_config()
    config_gen = generator_config()
    config_dis = discriminator_config()
    np.random.seed(config_train.seed)
    assert config_train.start_token == 0

    #Build dataloader for generaotr, testing and discriminator
    gen_data_loader = Gen_Data_loader(config_gen.gen_batch_size, config_gen.sequence_length)
    likelihood_data_loader = Gen_Data_loader(config_gen.gen_batch_size, config_gen.sequence_length)
    dis_data_loader = Dis_dataloader(config_dis.dis_batch_size, config_gen.sequence_length)

    #Build generator and its rollout
    generator = Generator(config=config_gen)
    generator.build()


    #Build target LSTM
    target_params = cPickle.load(open('save/reduced_target_params.pkl','rb'), encoding='iso-8859-1')
    target_lstm = TARGET_LSTM(config=config_gen, params=target_params) # The oracle model

    #Build discriminator
    discriminator = Discriminator(config=config_dis)
    discriminator.build_discriminator()

    #Build optimizer op for pretraining
    pretrained_optimizer = tf.train.AdamOptimizer(config_train.gen_learning_rate)
    var_pretrained = [v for v in tf.trainable_variables() if 'teller' in v.name] #Using name 'teller' here to prevent name collision of target LSTM
    gradients, variables = zip(*pretrained_optimizer.compute_gradients(generator.pretrained_loss, var_list=var_pretrained))
    gradients, _ = tf.clip_by_global_norm(gradients, config_train.grad_clip)
    gen_pre_upate = pretrained_optimizer.apply_gradients(zip(gradients, variables))

    #Initialize all variables
    sess = tf.Session(config=config_hardware)
    sess.run(tf.global_variables_initializer())

    #Initalize data loader of generator
    generate_samples(sess, target_lstm, config_train.batch_size, config_train.generated_num, config_train.positive_file)
    gen_data_loader.create_batches(config_train.positive_file)
    # bleu reward function
    reference_corpus = []
    with open(config_train.positive_file)as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            reference_corpus.append(parse_line)
    reference_length = 0
    reference_length += min(len(r) for r in reference_corpus)
    merged_ref_ngram_counts = collections.Counter()
    for reference in reference_corpus:
        merged_ref_ngram_counts |= bleu._get_ngrams(reference)
    def bleu_reward(sentence):
        return bleu.compute_bleu(reference_length, merged_ref_ngram_counts, sentence)[0]
    def bleu_reward_list(samples):
        d1, d2, d3, d4 = np.shape(samples)
        result = [[[bleu_reward(samples[i, j, k, :]) for k in range(d3)] for j in range(d2)] for i in range(d1)]
        return np.array(result)
    #Start pretraining
    print ('Start pre-training generator...')
    log = open('save/experiment-log.txt', 'w')
    log.write('pre-training...\n')
    epoch = 0
    for epoch in range(config_train.pretrained_epoch_num):
        gen_data_loader.reset_pointer()
        for it in range(gen_data_loader.num_batch):
            batch = gen_data_loader.next_batch()
            _, g_loss = sess.run([gen_pre_upate, generator.pretrained_loss], feed_dict={generator.input_seqs_pre:batch,\
                                                                                    generator.input_seqs_mask:np.ones_like(batch)})
        if epoch % config_train.test_per_epoch == 0:
            generate_samples(sess, generator, config_train.batch_size, config_train.generated_num, config_train.eval_file)
            likelihood_data_loader.create_batches(config_train.eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            #print ('pre-train epoch ', epoch, 'test_loss ', test_loss)
            #buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
            bleu_test_reward = bleu_reward_eval(sess, bleu_reward, likelihood_data_loader)
            buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(test_loss) + '\tbleu:\t' + str(bleu_test_reward) + '\n'
            print ('pre-train epoch ', epoch, 'test_loss: ', test_loss, 'bleu_test_reward: ', bleu_test_reward)
            log.write(buffer)
    print ('Start pre-training discriminator...')
    for t in range(config_train.dis_update_time_pre):
        print ("Times: " + str(t))
        generate_samples(sess, generator, config_train.batch_size, config_train.generated_num, config_train.negative_file)
        dis_data_loader.load_train_data(config_train.positive_file, config_train.negative_file)
        for _ in range(config_train.dis_update_epoch_pre):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: config_dis.dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)

    #Build optimizer op for adversarial training
    train_adv_opt = tf.train.AdamOptimizer(config_train.gen_learning_rate)
    gradients, variables = zip(*train_adv_opt.compute_gradients(generator.gen_loss_adv,var_list=var_pretrained))
    gradients, _ = tf.clip_by_global_norm(gradients, config_train.grad_clip)
    train_adv_update = train_adv_opt.apply_gradients(zip(gradients, variables))

    arm_grad = tf.gradients(generator.logits_list, var_pretrained, grad_ys=generator.arm_gradients)#/b_s
    train_arm_gradvars = zip(arm_grad, var_pretrained)
    train_arm_update = train_adv_opt.apply_gradients(train_arm_gradvars)
    #Initialize global variables of optimizer for adversarial training
    uninitialized_var = [e for e in tf.global_variables() if e not in tf.trainable_variables()]
    init_vars_uninit_op = tf.variables_initializer(uninitialized_var)
    sess.run(init_vars_uninit_op)

    # Start adversarial training
    if config_train.adversarial_training:
        rollout_gen = rollout(config=config_gen, full_sequence=True)
        print('Start adversarial training')
        for total_batch in range(config_train.total_batch):
            for iter_gen in range(config_train.gen_update_time):
                samples = sess.run(generator.sample_word_list_reshape)
                feed = {"pred_seq_rollout:0":samples}
                reward_rollout = []
                #calcuate the reward given in the specific stpe t by roll out
                for iter_roll in range(config_train.rollout_num):
                    rollout_list = sess.run(rollout_gen.sample_rollout_step, feed_dict=feed)
                    rollout_list_stack = np.vstack(rollout_list) #shape: #batch_size * #rollout_step, #sequence length
                    reward_rollout_seq = sess.run(discriminator.ypred_for_auc, feed_dict={discriminator.input_x:rollout_list_stack, discriminator.dropout_keep_prob:1.0})
                    reward_last_tok = sess.run(discriminator.ypred_for_auc, feed_dict={discriminator.input_x:samples, discriminator.dropout_keep_prob:1.0})
                    reward_allseq = np.concatenate((reward_rollout_seq, reward_last_tok), axis=0)[:,1]
                    reward_tmp = []
                    for r in range(config_gen.gen_batch_size):
                        reward_tmp.append(reward_allseq[range(r, config_gen.gen_batch_size * config_gen.sequence_length, config_gen.gen_batch_size)])
                    reward_rollout.append(np.array(reward_tmp))
                rewards = np.sum(reward_rollout, axis=0)/config_train.rollout_num
                _, gen_loss = sess.run([train_adv_update, generator.gen_loss_adv], feed_dict={generator.input_seqs_adv:samples,\
                                                                                            generator.rewards:rewards})
            if total_batch % config_train.test_per_epoch == 0 or total_batch == config_train.total_batch - 1:
                generate_samples(sess, generator, config_train.batch_size, config_train.generated_num, config_train.eval_file)
                likelihood_data_loader.create_batches(config_train.eval_file)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                bleu_test_reward = bleu_reward_eval(sess, bleu_reward, likelihood_data_loader)
                buffer = 'epoch:\t' + str(total_batch + epoch) + '\tnll:\t' + str(test_loss) + '\tbleu:\t' + str(bleu_test_reward) + '\n'
                print ('total_batch: ', total_batch + epoch, 'test_loss: ', test_loss, 'bleu_test_reward: ', bleu_test_reward)
                #buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
                #print ('total_batch: ', total_batch, 'test_loss: ', test_loss)
                log.write(buffer)

            for _ in range(config_train.dis_update_time_adv):
                generate_samples(sess, generator, config_train.batch_size, config_train.generated_num, config_train.negative_file)
                dis_data_loader.load_train_data(config_train.positive_file, config_train.negative_file)

                for _ in range(config_train.dis_update_epoch_adv):
                    dis_data_loader.reset_pointer()
                    for it in range(dis_data_loader.num_batch):
                        x_batch, y_batch = dis_data_loader.next_batch()
                        feed = {
                            discriminator.input_x: x_batch,
                            discriminator.input_y: y_batch,
                            discriminator.dropout_keep_prob: config_dis.dis_dropout_keep_prob
                        }
                        _ = sess.run(discriminator.train_op, feed)
    # start training with n-gram loss
    elif config_train.n_gram_training:
        print('Start bleu training')
        # begin training
        if config_train.reinforce:
            print("Training with REINFORCE")
            rollout_gen = rollout(config=config_gen, full_sequence=True)
            for total_batch in range(config_train.total_batch):
                for iter_gen in range(config_train.gen_update_time):
                    samples = sess.run(generator.sample_word_list_reshape)
                    feed = {"pred_seq_rollout:0":samples}
                    #reward_rollout = []
                    #calcuate the reward given in the specific stpe t by roll out
                    #for iter_roll in range(config_train.rollout_num):
                    rollout_list = sess.run(rollout_gen.sample_rollout_step, feed_dict=feed)
                    #print(rollout_list)
                    reward_rollout_seq = []
                    # rollout_list_stack = np.vstack(rollout_list) #shape: #batch_size * #rollout_step, #sequence length
                    for tmp in rollout_list:
                        #reward_rollout_seq += [bleu_reward(sentence) for sentence in tmp]
                        reward_rollout_seq += [bleu_reward(sentence) for sentence in samples]
                    reward_last_tok = [bleu_reward(sentence) for sentence in samples]
                    reward_allseq = np.concatenate((reward_rollout_seq, reward_last_tok), axis=0)    # sequene_length * batch_size
                    reward_tmp = []
                    for r in range(config_gen.gen_batch_size):
                        reward_tmp.append(reward_allseq[range(r, config_gen.gen_batch_size * config_gen.sequence_length, config_gen.gen_batch_size)])
                    #reward_rollout.append(np.array(reward_tmp))
                    #rewards = np.sum(reward_rollout, axis=0)/config_train.rollout_num
                    rewards = np.array(reward_tmp)
                    _, gen_loss = sess.run([train_adv_update, generator.gen_loss_adv], feed_dict={generator.input_seqs_adv:samples,\
                                                                                                generator.rewards:rewards})
                if total_batch % config_train.test_per_epoch == 0 or total_batch == config_train.total_batch - 1:
                    generate_samples(sess, generator, config_train.batch_size, config_train.generated_num, config_train.eval_file)
                    likelihood_data_loader.create_batches(config_train.eval_file)
                    test_loss = target_loss(sess, target_lstm, likelihood_data_loader)

                    bleu_test_reward = bleu_reward_eval(sess, bleu_reward, likelihood_data_loader)
                    buffer = 'epoch:\t' + str(total_batch + epoch) + '\tnll:\t' + str(test_loss) + '\tbleu:\t' + str(bleu_test_reward) + '\n'
                    print ('total_batch: ', total_batch + epoch, 'test_loss: ', test_loss, 'bleu_test_reward: ', bleu_test_reward)
                    log.write(buffer)
                    #buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
                    #print ('total_batch: ', total_batch, 'test_loss: ', test_loss)
                    #log.write(buffer)
        elif config_train.arm:
            print("Training with ARM")
            for total_batch in range(config_train.total_batch):
                for iter_gen in range(config_train.gen_update_time):
                    samples, pi_samples = sess.run((generator.sample_arm_word_list_reshape, generator.pi_sample_list))
                    #reward_rollout = []
                    #calcuate the reward given in the specific stpe t by roll out
                    #for iter_roll in range(config_train.rollout_num):
                    rewards_list = bleu_reward_list(samples) # sequence_length * batch_size * num_emb
                    _ = sess.run([train_arm_update], feed_dict={generator.pi_sample_input: pi_samples, generator.rewards_list:rewards_list})
                if total_batch % config_train.test_per_epoch == 0 or total_batch == config_train.total_batch - 1:
                    generate_samples(sess, generator, config_train.batch_size, config_train.generated_num, config_train.eval_file)
                    likelihood_data_loader.create_batches(config_train.eval_file)
                    test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                    # buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
                    # print ('total_batch: ', total_batch, 'test_loss: ', test_loss)
                    bleu_test_reward = bleu_reward_eval(sess, bleu_reward, likelihood_data_loader)
                    buffer = 'epoch:\t' + str(total_batch + epoch) + '\tnll:\t' + str(test_loss) + '\tbleu:\t' + str(bleu_test_reward) + '\n'
                    print ('total_batch: ', total_batch + epoch, 'test_loss: ', test_loss, 'bleu_test_reward: ', bleu_test_reward)
                    log.write(buffer)
    log.close()
if __name__ == "__main__":
    tf.app.run()
