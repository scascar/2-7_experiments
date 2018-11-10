from treys import Card, Evaluator, Deck
import random
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model

# input: 52+9 classes
# output: 32


def load_trained_model(path):
    return load_model(path)


def make_model():
    input_all = Input(shape=(61,), name="input")
    x = Dense(128, activation='relu')(input_all)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(32, activation='relu')(x)
    model = Model(inputs=[input_all], outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

# hand: 5 Card
# change_combinaison: int 0-31 mapped to [X X X X X]


def map_to_draw(combinaison):
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


'''
    High Card      + 1277     [(13 choose 5) - 10 straights]
    -------------------------
    TOTAL            7462
    6185
'''


def get_reward(new_eval):
    if new_eval < 6185:
        return 0
    else:
        return (new_eval-6185)/1277

# Get the feature vector of (,52)


def hand_to_indexes(hand):
    feature = np.zeros(52)
    for card in hand:
        feature[card_to_total_index(card)] = 1

    return feature


def hand_to_rank_class(hand):
    classes = np.zeros(9)
    classes[evaluator.get_rank_class(evaluator.evaluate([], hand))-1] = 1
    return classes


def draw(hand, change_combinaison):
    mapped_combinaison = map_to_draw(change_combinaison)
    for i in range(5):
        if mapped_combinaison[i] == 1:
            hand[i] = deck.draw(1)


# Means all possible draws to even out variance
def draw_until_empty_reward_mean(hand, change_combinaison):
    draws_count = 0
    total_score = 0
    if change_combinaison != 0:
        # We can still full draw
        while len(deck.cards) > 5:
            draws_count += 1
            draw(hand, change_combinaison)
            total_score += evaluator.evaluate([], hand)

        return int(total_score/draws_count)
    else:
        return evaluator.evaluate([], hand)


deck = Deck()
evaluator = Evaluator()
model = load_trained_model("models/modelremade.h5")
#model = make_model()
num_hands = 150000
train_steps = 10000
max_replay = 10000
epsi = 0.5
metrics_window = 1000

train_X = []
train_Y = []

metrics_score = []
current_metric_score = 0
for i in range(num_hands):

    deck.shuffle()
    hand = deck.draw(5)
    if (i+1) % train_steps == 0:
        old_hand = hand.copy()

    feat = np.concatenate([hand_to_indexes(hand), hand_to_rank_class(hand)])
    Q = model.predict(feat.reshape(1, 61))
    if random.random() < epsi:
        action = np.argmax(Q)
    else:
        action = random.randint(0, 31)

    new_eval = draw_until_empty_reward_mean(hand, action)
    reward = get_reward(new_eval)
    current_metric_score += reward
    if (i+1) % metrics_window == 0:
        metrics_score.append(current_metric_score/metrics_window)
        current_metric_score = 0
    train_X.append(feat.reshape((61,)))

    Q[0, action] = reward
    train_Y.append(Q.reshape((32,)))
    if (i+1) % train_steps == 0:
        print("hand n° ", str(i))
        print("random %", str(epsi))
        print(Card.print_pretty_cards(old_hand))
        print(map_to_draw(np.argmax(Q)))
        print(Card.print_pretty_cards(hand))
        print(Q)
        model.fit(np.stack(train_X), np.stack(train_Y), verbose=1, epochs=5)

        # reduce explo
        epsi *= 0.9

    if len(train_X) > max_replay:
        train_X = train_X[50:]
        train_Y = train_Y[50:]

print(metrics_score)
print("Saving model..")
model.save('models/modelremade.h5')
print("Model saved to models/modelremade.h5..")
