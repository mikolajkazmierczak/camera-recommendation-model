import random
import numpy as np
from numpy.linalg import norm


NUM_USERS = 7
NUM_CAMERAS = 4
NUM_COMPONENTS = 10
REVIEW_VALUES = [1, 0, -1, None]
CAMERA_REVIEW_PROBABILITY = 0.8


def gen_db():
    users = []
    for _ in range(NUM_USERS):
        user = []
        for _ in range(NUM_CAMERAS):
            camera = [None, None, None, None]
            if random.random() < CAMERA_REVIEW_PROBABILITY:
                camera = [random.choice(REVIEW_VALUES) for _ in range(NUM_COMPONENTS)]
            user.append(camera)
        users.append(user)
    return users


def aggregate_db(db, log=False):
    aggregated_db = []
    for user in db:
        # count how many times each component is reviewed in any way
        count_reviewed_components_per_camera = [0 for _ in range(NUM_CAMERAS)]
        for i, camera in enumerate(user):
            for k in camera:
                if k is not None:
                    count_reviewed_components_per_camera[i] += 1
        # assign a weight to each component
        weights = [c / NUM_COMPONENTS for c in count_reviewed_components_per_camera]
        if log:
            print(weights, end=" | ")
        # normalize weights (0-1 that sum to 1)
        normalized = [w / sum(weights) for w in weights]
        if log:
            print(normalized)
        # assign aggregated values (a = w1*k1 + w2*k2 + ... + wn*kn)
        values = []
        for i, camera in enumerate(user):
            a = 0
            for k in camera:
                if k is not None:
                    a += normalized[i] * k
            values.append(a)
        aggregated_db.append(values)
    return aggregated_db


def promotor(value):
    return value**2


def normalize_aggregated_db(db, promote=True):
    for user in db:
        # add min value to all values (so that all values are positive)
        min_a = min(user)
        for i, a in enumerate(user):
            user[i] += abs(min_a)
    # apply promotor function to all values
    if promote:
        for user in db:
            for i, a in enumerate(user):
                user[i] = promotor(a)
    # normalize values (0-1 that sum to 1)
    db = [
        [a / sum(user) for a in user] if sum(user) != 0 else [0 for _ in user]
        for user in db
    ]
    return db


def print_db(db):
    for i, user in enumerate(db):
        print(f"u{i} {user}")
        print(f"u{i} a0 a1 a2 a3")
        for ki in range(NUM_COMPONENTS):
            print(f"k{ki}", end="")
            for j, camera in enumerate(user):
                k = camera[ki]
                if k is None:
                    print(f"  .", end="")
                elif k is -1:
                    print(f" -1", end="")
                else:
                    print(f"  {k}", end="")
            print()


def print_aggregated_db(db):
    contains_negative_numbers = [[a < 0 for a in user] for user in db]
    print(f"    a0    a1    a2    a3")
    for i, user in enumerate(db):
        print(f"u{i}", end="")
        for j, a in enumerate(user):
            if a > 0 and contains_negative_numbers:
                print(f"  {a:.2f}", end="")
            elif a == 0:
                print(f"  0   ", end="")
            else:
                print(f" {a:.2f}", end="")
        print()


def cosine(u1, u2):
    u1, u2 = np.array(u1), np.array(u2)
    return np.dot(u1, u2) / (norm(u1) * norm(u2))


def recommend(db, user_i):
    user = db[user_i]
    # find most similar user
    best_cosine = -1
    best_user_i = -1
    for i, other_user in enumerate(db):
        if i == user_i:
            continue
        val = cosine(user, other_user)
        if val > best_cosine:
            best_cosine = val
            best_user_i = i
    # find most recommended camera
    best_user = db[best_user_i]
    camera_i = np.argmax(best_user)
    return camera_i, best_user_i


from sample import db

# db = gen_db()

print("=== Database:")
print_db(db)

print("=== Aggregated database:")
aggregated_db = aggregate_db(db)
print_aggregated_db(aggregated_db)

normalized_aggregated_db = normalize_aggregated_db(aggregated_db, promote=True)
print("=== Normalized aggregated database:")
print_aggregated_db(normalized_aggregated_db)

r = recommend(normalized_aggregated_db, 0)
print(f"Recommended for u0: a{r[0]} <- u{r[1]} ")
