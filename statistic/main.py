import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def get_emotion_change(qmasks, labels):
    male_emotion = []
    female_emotion = []
    result = []
    count_diff = 0
    count_same = 0
    for conv in labels:
        for i in range(len(labels[conv])):
            if qmasks[conv][i] == 'M':
                if len(male_emotion) < 2:
                    male_emotion.append(labels[conv][i])
                    if len(male_emotion) == 2 and male_emotion[0] != male_emotion[1]:
                        count_diff += 1
                        if female_emotion:
                            result.append([male_emotion[0], male_emotion[1], female_emotion[-1]])
                        else:
                            result.append([male_emotion, male_emotion[1], None])
                    elif len(male_emotion) == 2 and male_emotion[0] == male_emotion[1]:
                        count_same += 1
                else:
                    male_emotion[0] = male_emotion[1]
                    male_emotion[1] = labels[conv][i]
                    if male_emotion[0] != male_emotion[1]:
                        count_diff += 1
                        if female_emotion:
                            result.append([male_emotion[0], male_emotion[1], female_emotion[-1]])
                        else:
                            result.append([male_emotion, male_emotion[1], None])
                    else:
                        count_same += 1
            if qmasks[conv][i] == 'F':
                if len(female_emotion) < 2:
                    female_emotion.append(labels[conv][i])
                    if len(female_emotion) == 2 and female_emotion[0] != female_emotion[1]:
                        count_diff += 1
                        if male_emotion:
                            result.append([female_emotion[0], female_emotion[1], male_emotion[-1]])
                        else:
                            result.append([female_emotion, female_emotion[1], None])
                    elif len(female_emotion) == 2 and female_emotion[0] == female_emotion[1]:
                        count_same += 1
                else:
                    female_emotion[0] = female_emotion[1]
                    female_emotion[1] = labels[conv][i]
                    if female_emotion[0] != female_emotion[1]:
                        count_diff += 1
                        if male_emotion:
                            result.append([female_emotion[0], female_emotion[1], male_emotion[-1]])
                        else:
                            result.append([female_emotion, female_emotion[1], None])
                    else:
                        count_same += 1
    for i in result:
        assert len(i) == 3
    return result, count_diff / (count_same + count_diff)


def get_emotion_dict(emotion_change):
    emotion_dict = {}
    label_mapping = {0: 'hap', 1: 'sad', 2: 'neu', 3: 'ang', 4: 'exc', 5: 'fru'}
    count = 0
    for i in range(6):
        for j in range(6):
            if i != j:
                string = label_mapping[i] + " to " + label_mapping[j]
                emotion_dict[string] = 0
    for i in emotion_change:
        string = label_mapping[i[0]] + " to " + label_mapping[i[1]]
        emotion_dict[string] += 1
        if i[1] == i[2]:
            count += 1
    p = count / len(emotion_change)
    return emotion_dict, p


def get_bar_chart(emotion_dict):
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({'figure.autolayout': True})
    data_list = [{k: v} for k, v in emotion_dict.items()]
    f = lambda x: list(x.values())[0]
    emotion_list = sorted(data_list, key=f, reverse=False)
    group_data = [list(i.values())[0] for i in emotion_list]
    group_names = [list(i.keys())[0] for i in emotion_list]
    group_mean = np.mean(group_data)
    fig, ax = plt.subplots(figsize=(16, 32))
    ax.barh(group_names, group_data)
    ax.set(xlabel='Frequency', ylabel='Change Category', title='Emotion Change')
    ax.axvline(group_mean, ls='--', color='r')
    plt.show()


def get_pie_chart(labels):
    emotion_dict = {}
    label_mapping = {0: 'happy', 1: 'sad', 2: 'neutral', 3: 'anger', 4: 'excited', 5: 'frustrated'}
    for i in range(6):
        emotion_dict[label_mapping[i]] = 0
    for conv in labels:
        for i in range(len(labels[conv])):
            emotion_dict[label_mapping[labels[conv][i]]] += 1
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({'figure.autolayout': True})
    group_data = list(emotion_dict.values())
    group_names = list(emotion_dict.keys())
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(group_data, labels=group_names, autopct='%0.1f%%', shadow=True, startangle=90)
    ax.axis('equal')
    ax.set(title='Emotion Distribution')
    plt.show()


if __name__ == '__main__':
    _, _, qmasks, labels, _, _ = pickle.load(open("IEMOCAP_features_bert.pkl", "rb"), encoding="latin1")
    emotion_change, p_change = get_emotion_change(qmasks, labels)
    emotion_dict, p_same = get_emotion_dict(emotion_change)
    get_bar_chart(emotion_dict)
    get_pie_chart(labels)