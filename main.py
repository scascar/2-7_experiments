from treys import Card, Evaluator, Deck
import random
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model

# input: 52
# output: 32


def make_model():
    input_all = Input(shape=(52,), name="input")
    x = Dense(256, activation='relu')(input_all)
    x = Dense(512, activation='relu')(x)
    x = Dense(256)(x)
    out = Dense(32)(x)
    model = Model(inputs=[input_all], outputs=out)
    model.compile(optimizer='rmsprop', loss='mse')
    return model

# hand: 5 Card
# change_combinaison: int 0-31 mapped to [X X X X X]


def __map_to_draw(combinaison):
    vec = [int(x) for x in list('{0:0b}'.format(combinaison))]
    while len(vec) < 5:
        vec.insert(0, 0)
    return vec


def card_to_total_index(c):
    suit = Card.get_suit_int(c)
    # keeping 1 as is
    if suit == 2:
        suit = 0
    elif suit == 4:
        suit = 2
    elif suit == 8:
        suit = 3

    return Card.get_rank_int(c)*(suit+1)


def get_reward(old_eval, new_eval):
    # highest eval: best hand in deuce
    # ratio > 1 : improvement
    ratio = new_eval/old_eval
    return ratio

# Get the feature vector of (,52)


def hand_to_indexes(hand):
    feature = np.zeros(52)
    for card in hand:
        feature[card_to_total_index(card)] = 1

    return feature


def draw(hand, change_combinaison):
    mapped_combinaison = __map_to_draw(change_combinaison)
    for i in range(5):
        if mapped_combinaison[i] == 1:
            hand[i] = deck.draw(1)


deck = Deck()
evaluator = Evaluator()
model = make_model()
num_hands = 50000
train_steps = 2000
max_replay = 5000
epsi = 0.05

train_X = []
train_Y = []

for i in range(num_hands):

    deck.shuffle()
    hand = deck.draw(5)
    old_eval = evaluator.evaluate([], hand)

    feat = hand_to_indexes(hand)
    if random.random() > epsi:
        Q = model.predict(feat.reshape(1, 52))
        action = np.argmax(Q)
    else:
        action = random.randint(0, 31)

    draw(hand, action)
    new_eval = evaluator.evaluate([], hand)

    reward = get_reward(old_eval, new_eval)
    train_X.append(feat.reshape((52,)))

    Q[0, action] = reward
    train_Y.append(Q.reshape((32,)))

    if (i+1) % train_steps == 0:
        print("hand n° ", str(i))
        print(Card.print_pretty_cards(hand))
        model.fit(np.stack(train_X), np.stack(train_Y), verbose=1, epochs=5)

    if len(train_X) > max_replay:
        train_X = train_X[50:]
        train_Y = train_Y[50:]
