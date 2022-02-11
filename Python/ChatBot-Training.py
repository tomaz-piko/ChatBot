#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
#import re
import numpy as np
import tensorflow as tf
import datetime
import pathlib
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(42)
from tensorflow.keras.utils import plot_model


# In[2]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
get_ipython().run_line_magic('load_ext', 'tensorboard')
#%reload_ext tensorboard


# # Reading lines from processed files

# In[3]:


def lines_eng(max_line_length = 999):
    q_filtered = []
    a_filtered = []
    fileQ = open('./DataENG/questions.txt', 'r', encoding='utf8')
    fileA = open('./DataENG/answers.txt', 'r', encoding='utf8')
    q = fileQ.read().split('\n')
    a = fileA.read().split('\n')
    for i in range(len(q)):
        if i >= len(a):
            break
        line1 = q[i]
        line2 = '<BOS> ' + a[i] + ' <EOS>'
        if len(line1.split(' ')) > (max_line_length):
            continue
        elif len(line2.split(' ')) > (max_line_length + 2):
            continue
        else:
            q_filtered.append(line1)
            a_filtered.append(line2)
    fileQ.close()
    fileA.close()
    return q_filtered, a_filtered

def lines_slo(max_line_length = 999):
    q_filtered = []
    a_filtered = []
    file = open('./DataSLO/Processed_Final3.txt', 'r', encoding='utf8')
    lines = file.read().split('\n')
    myIter = iter(range(0, len(lines)- 1, 2))
    for i in myIter:
        line1 = lines[i]
        line2 = '<START> ' + lines[i+1] + ' <END>'
        if len(line1.split(' ')) > (max_line_length):
            continue
        elif len(line2.split(' ')) > (max_line_length + 2):
            continue
        else:
            q_filtered.append(line1)
            a_filtered.append(line2)
    file.close()
    return q_filtered, a_filtered
        
def create_vocab(tokenizer, max_vocab_size):
    word2idx = {}
    idx2word = {}
    for k, v in tokenizer.word_index.items():
        if v < max_vocab_size: 
            word2idx[k] = v
            idx2word[v] = k
        else: 
            break
    return word2idx, idx2word


# # Inputing parameters

# In[4]:


LANG = input('Language ENG/SLO: ')
EPOCHS = int(input('Epochs: '))
BATCH_SIZE = int(input('Batch size: '))
#EMBEDD_SIZE = int(input('Embedd size: '))
EMBEDD_SIZE = 100
#LATENT_DIM = int(input('Latent dim: '))
LATENT_DIM = 128
if LANG == 'ENG':
    MAX_LEN = 30
else:
    MAX_LEN = 15
#NUM_LINES = int(input('Max number of lines: '))
NUM_LINES = 25000
#VOCAB_SIZE = int(input('Vocab size: '))
VOCAB_SIZE = 15000

save_folder_path = LANG + str(EPOCHS) + '_BS' + str(BATCH_SIZE) + '_LD' + str(LATENT_DIM) + '_ES' + str(EMBEDD_SIZE) + '_LC' + str(NUM_LINES/1000) + 'k_VS' + str(VOCAB_SIZE)


# # Preparing inputs, outputs & vocabulary

# In[5]:


if LANG == 'ENG':
    encoder_text, decoder_text = lines_eng(max_line_length=MAX_LEN)
elif LANG == 'SLO':
    encoder_text, decoder_text = lines_slo(max_line_length=MAX_LEN)
if NUM_LINES > 0 and NUM_LINES < len(encoder_text):
    encoder_text = encoder_text[:NUM_LINES]
    decoder_text = decoder_text[:NUM_LINES]


# In[6]:


print(encoder_text[0] + " --> " + decoder_text[0])
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(encoder_text + decoder_text)
word2idx, idx2word = create_vocab(tokenizer, VOCAB_SIZE)
print("Word2idx length: ", str(len(word2idx)))
print(len(word2idx) == len(idx2word))
encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
decoder_sequences = tokenizer.texts_to_sequences(decoder_text)
SEQUENCE_COUNT = len(encoder_sequences)*2
print("Sequence count: ", SEQUENCE_COUNT)
print(len(encoder_sequences) == len(decoder_sequences))
VOCAB_SIZE = len(idx2word) + 1
print("Vocab size: ", VOCAB_SIZE)
print(str(encoder_sequences[0]) + " --> " + str(decoder_sequences[0]))


# In[7]:


encoder_max_length = max([len(l.split()) for l in encoder_text])
decoder_max_length = max([len(l.split()) for l in decoder_text])
print("Encoder max length: " + str(encoder_max_length))
print("Decoder max length: " + str(decoder_max_length))
MAX_LEN = decoder_max_length


# In[8]:


def generate_inputs(encoder_sequences, decoder_sequences):
    num_batches = len(encoder_sequences) // BATCH_SIZE
    while True:
        for i in range(0, num_batches):
            start = i * BATCH_SIZE
            end = (i+1) * BATCH_SIZE
            encoder_input_data = pad_sequences(encoder_sequences[start:end:1],
                                               maxlen=encoder_max_length,
                                               dtype='int32',
                                               padding='post',
                                               truncating='post')
            decoder_input_data = pad_sequences(decoder_sequences[start:end:1],
                                               maxlen=decoder_max_length,
                                               dtype='int32',
                                               padding='post',
                                               truncating='post')
            decoder_output_data = np.zeros((BATCH_SIZE, decoder_max_length, VOCAB_SIZE), dtype='float32')
            for j, seqs in enumerate(decoder_input_data):
                for k, seq in enumerate(seqs):
                    if k > 0:
                        decoder_output_data[j][k - 1][seq] = 1.
            yield [encoder_input_data, decoder_input_data], decoder_output_data
            
Xtrain, Xtest, Ytrain, Ytest = train_test_split(encoder_sequences, decoder_sequences, test_size=0.2, random_state=42)
train_gen = generate_inputs(Xtrain, Ytrain)
test_gen = generate_inputs(Xtest, Ytest)
train_gen_batches = len(Xtrain) // BATCH_SIZE
test_gen_batches = len(Xtest) // BATCH_SIZE           


# # Loading embedding & creating shared embedding layer

# In[9]:


def embedding_matrix(word2em, word2idx, embedd_size):
    embedding_matrix = np.zeros((len(word2idx) + 1, embedd_size))
    #fileW = open('./DataSLO/unknown_tokens.txt', 'w', encoding='utf-8')
    unknown = 0
    for word, i in word2idx.items():
        embedding_vector = word2em.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            #fileW.write(word + '\n')
            unknown += 1  
    print('Embedding matrix UNK token count: ', str(unknown))
    return embedding_matrix

def load_glove_embedding(embedd_size=100):
    word2em = {}
    file = open('./DataENG/glove.6B.' + str(embedd_size) + 'd.txt', 'r', encoding='utf8')
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        word2em[word] = vector
    file.close()
    print("Glove loaded...")
    print("Embedding size: ", str(embedd_size))
    return word2em, embedd_size

def load_word2vec_embedding(embedd_size=100):
    word2em = {}
    if VOCAB_SIZE > 10000:
        file = open('./DataSLO/embedding_SLO_' + str(embedd_size) + '.txt', 'r', encoding='utf8')
    else:
        file = open('./DataSLO/embedding_SLO_' + str(embedd_size) + '_2.txt', 'r', encoding='utf8')
    lines = file.readlines()
    data_line = lines[0].split()
    embedd_size = int(data_line[1])
    for line in lines[1:]:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        word2em[word] = vector
    file.close()
    print("Word2vec loaded...")
    print("Embedding size: ", str(embedd_size))
    return word2em, embedd_size   
#EMBEDD_SIZE=200
if LANG == 'ENG':
    word2em, embedd_size = load_glove_embedding(embedd_size=EMBEDD_SIZE)
elif LANG == 'SLO':
    word2em, embedd_size = load_word2vec_embedding(embedd_size=EMBEDD_SIZE)
embedding_matrix = embedding_matrix(word2em, word2idx, embedd_size)
word2em = []
decoder_text = []
embedding_layer = Embedding(input_dim=VOCAB_SIZE,
                            output_dim=embedd_size,
                            weights=[embedding_matrix],
                            trainable=True)
print(embedding_matrix)


# # Creating model

# In[10]:


OPT = Adam()
encoder_inputs = Input(shape=(None, ), dtype='int32', name='encoder_inputs')
encoder_embedding = embedding_layer(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(units=LATENT_DIM,
                                         return_state=True,
                                         name='encoder_LSTM'
                                        )(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, ), dtype='int32', name='decoder_inputs')
decoder_embedding = embedding_layer(decoder_inputs)
decoder_LSTM = LSTM(units=LATENT_DIM,
                    return_state=True,
                    return_sequences=True,
                    name='decoder_LSTM')
decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=encoder_states)

decoder_dense = Dense(VOCAB_SIZE, activation='softmax', name='decoder_dense')
outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], outputs)
model.compile(loss='categorical_crossentropy', optimizer=OPT, metrics=['accuracy'])

model.summary()


# In[12]:


plot_model(model, to_file='model_plot.pdf', show_shapes=True, show_layer_names=True)


# # Model training

# In[35]:


log_dir = "logs\\fit\\" + save_folder_path
callbacks = [TensorBoard(log_dir=log_dir)]
history = model.fit(x=train_gen,
                    steps_per_epoch=train_gen_batches,
                    epochs=EPOCHS,
                    verbose=1, 
                    validation_data=test_gen,
                    validation_steps=test_gen_batches,
                    callbacks=callbacks
                   )


# ### Model training data plots

# In[36]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('model accuracy')
ax1.set(xlabel='epoch', ylabel = 'accuracy')
ax1.legend(['train_accuracy', 'val_accuracy'], loc='upper left')
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set(xlabel='epoch', ylabel='loss')
ax2.legend(['train_loss', 'val_loss'], loc='upper left')
plt.show()


# ### Creating inference models

# In[37]:


encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(LATENT_DIM, ))
decoder_state_input_c = Input(shape=(LATENT_DIM, ))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_LSTM(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


# In[40]:


plot_model(encoder_model, to_file='encoder_plot.png', show_shapes=True, show_layer_names=True)


# In[39]:


plot_model(decoder_model, to_file='decoder_plot.png', show_shapes=True, show_layer_names=True)


# ### Model testing

# In[14]:


def decode_sequence(input_seq):
    if LANG == 'ENG':
        start_token = 'bos'
        end_token = 'eos'
    else:
        start_token = 'start'
        end_token = 'end'
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros(( 1, 1), dtype='int32')
    target_seq[0, 0] = word2idx[start_token]
    stop_condition = False
    decoded_sentence = ''
    word_count = 0
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])  
        sampled_word = idx2word[sampled_token_index]
        if not sampled_word == start_token and not sampled_word == end_token:
            decoded_sentence += sampled_word + " "
            word_count += 1
        if sampled_word == end_token or word_count >= MAX_LEN:
            stop_condition = True
        target_seq = np.zeros(( 1, 1), dtype='int32')
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return decoded_sentence


# In[16]:


from random import randint
rand = randint(0, 1000)
for i in range(rand, rand+5, 1):
    input_seq = np.zeros((1, MAX_LEN), dtype='int32') 
    line_list = encoder_sequences[i]
    index = 0
    for word in line_list: #seqToDecoderData
        input_seq[0, index] = word
        index += 1
    print('Input:  ' + encoder_text[i])
    response = decode_sequence(input_seq)
    print('Response:  ' + response)
    print()


# In[22]:


for i in range(5):
    index = 0
    line = input('Input:  ')
    input_seq = np.zeros((1, MAX_LEN), dtype='int32')    
    line_list = line.split(' ')
    for word in line_list:
        word = word.lower()
        if word in word2idx:
            input_seq[0, index] = word2idx[word]
            index += 1
    response = decode_sequence(input_seq)   
    print('Response:  ' + response)


# In[94]:


index = 0
line = input('Input:  ')
input_seq = np.zeros((1, MAX_LEN), dtype='int32')    
line_list = line.split(' ')
for word in line_list:
    word = word.lower()
    if word in word2idx:
        input_seq[0, index] = word2idx[word]
        index += 1
response = decode_sequence(input_seq)   
print('Response:  ' + response)


# # Model and data saving

# In[17]:


#save_folder_path = '200e_128bs_128dim_lrReduce_01drop'
try:
    pathlib.Path('./' + save_folder_path).mkdir(exist_ok=False)
except:
    save_folder_path = save_folder_path + '_1'
    pathlib.Path('./' + save_folder_path).mkdir(exist_ok=False)
    pass

saveData = dict()
saveData['vocab_size'] = VOCAB_SIZE
saveData['seq_count'] = SEQUENCE_COUNT
saveData['enc_max_length'] = MAX_LEN
saveData['dec_max_length'] = MAX_LEN
saveData['language'] = LANG
saveData['save_folder_path'] = save_folder_path
saveData['epochs'] = EPOCHS
saveData['latent_dim'] = LATENT_DIM
saveData['batch_size'] = BATCH_SIZE
saveData['optimizer'] = 'ADAM'

encoder_model.save('./' + save_folder_path + '/encoder.h5')
decoder_model.save('./' + save_folder_path + '/decoder.h5')
np.save('./' + save_folder_path + '/' + 'history.npy', history.history)


# In[18]:


print('CONVERTER COMMANDS')
print('tensorflowjs_converter --input_format keras Diplomska/' + save_folder_path + '/encoder.h5 Diplomska/' + save_folder_path + '/Encoder_tf')
print('tensorflowjs_converter --input_format keras Diplomska/' + save_folder_path + '/decoder.h5 Diplomska/' + save_folder_path + '/Decoder_tf')
fileW = open('./' + save_folder_path + '/converter_commands.txt', 'w')
fileW.write('tensorflowjs_converter --input_format keras Diplomska/' + save_folder_path + '/encoder.h5 Diplomska/' + save_folder_path + '/Encoder_tf' + '\n')
fileW.write('tensorflowjs_converter --input_format keras Diplomska/' + save_folder_path + '/decoder.h5 Diplomska/' + save_folder_path + '/Decoder_tf' + '\n')
fileW.close()

#Shranjevanje slovarjev in informacij o modelu v JSON datoteke.
with open('./' + save_folder_path + '/data.json', 'w') as fp:
    json.dump(saveData, fp)
with open('./' + save_folder_path + '/dictionary.json', 'w') as fp:
    json.dump(word2idx, fp)
with open('./' + save_folder_path + '/revDictionary.json', 'w') as fp:
    json.dump(idx2word, fp)
    
#Shranjevanje Keras modelov.
encoder_model.save('./' + save_folder_path + '/encoder.h5')
decoder_model.save('./' + save_folder_path + '/decoder.h5')


# In[25]:


print(word2idx['ja'])
print(idx2word[29])


# In[29]:


path = 'Not needed models/ENG_BS256_LD128_ES100_LC75.0k_VS15000'
history = np.load(path + '/history.npy',allow_pickle='TRUE').item()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(history['accuracy'])
ax1.plot(history['val_accuracy'])
ax1.set_title('NATANČNOST MODELA')
ax1.set(xlabel='epoch', ylabel = 'accuracy')
ax1.legend(['train_accuracy', 'val_accuracy'], loc='upper left')
ax2.plot(history['loss'])
ax2.plot(history['val_loss'])
ax2.set_title('IZGUBA MODELA')
ax2.set(xlabel='epoch', ylabel='loss')
ax2.legend(['train_loss', 'val_loss'], loc='upper left')
plt.show()


# In[28]:


paths = ['./SLO50_BS32_LD128_ES100_LC25.0k_VS15000_1', './ENG50_BS32_LD128_ES100_LC25.0k_VS15000']
histories = []
for path in paths:
    history = np.load(path + '/history.npy',allow_pickle='TRUE').item()
    histories.append(history)
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
for history in histories:    
    ax1.plot(history['accuracy'])
    ax1.plot(history['val_accuracy'])
    ax1.set_title('NATANČNOSTI MODELOV')
    ax1.set(xlabel='epoch', ylabel = 'accuracy')
    ax1.legend(
        ['SLO train_accuracy',
         'SLO val_accuracy',
         'ENG train_accuracy',
         'ENG val_accuracy'
        ], loc='upper left')
    ax2.plot(history['loss'])
    ax2.plot(history['val_loss'])
    ax2.set_title('IZGUBE MODELOV')
    ax2.set(xlabel='epoch', ylabel='loss')
    ax2.legend(
        ['SLO train_loss',
         'SLO val_loss',
         'ENG train_loss',
         'ENG val_loss'
        ], loc='upper left')

plt.show()


# In[6]:


paths = ['./SLO50_BS16_LD128_ES100_LC25.0k_VS15000',
         './SLO50_BS32_LD128_ES100_LC25.0k_VS15000',
         './SLO_BS64_LD128_ES100_LC25.0k_VS15000',
         './SLO_BS128_LD128_ES100_LC25.0k_VS15000'
        ]
histories = []
for path in paths:
    history = np.load(path + '/history.npy',allow_pickle='TRUE').item()
    histories.append(history)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
for history in histories:
    ax1.plot(history['accuracy'])
    ax2.plot(history['val_accuracy'])
    ax1.set_title('NATANČNOST MODELOV NA UČNI MNOŽICI')
    ax2.set_title('NATANČNOST MODELOV NA TESTNI MNOŽICI')
    ax1.set(xlabel='epoch', ylabel = 'train_accuracy')
    ax2.set(xlabel='epoch', ylabel = 'val_accuracy')

for history in histories:
    ax3.plot(history['loss'])
    ax4.plot(history['val_loss'])
    ax3.set_title('IZGUBE MODELOV NA UČNI MNOŽICI')
    ax4.set_title('IZGUBE MODELOV NA TESTNI MNOŽICI')
    ax3.set(xlabel='epoch', ylabel='train_loss')
    ax4.set(xlabel='epoch', ylabel='val_loss')

param = 'velikost paketa'
params = [16, 32, 64, 128]
legend_params = []
for p in params:
    legend_params.append(param + ' = ' + str(p))
    
plt.legend(legend_params,
          bbox_to_anchor=(1.42, 2))
plt.show()


# In[ ]:




