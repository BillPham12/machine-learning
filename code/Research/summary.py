from matplotlib import pyplot as plt
import numpy as np
adjective_list = [0.5059523809523809, 0.6584249084249084, 0.8315018315018315, 0.8690476190476191, 0.8704212454212454, 0.8722527472527473, 0.8777472527472527, 0.8791208791208791, 0.8745421245421245, 0.8695054945054945, 0.8667582417582418, 0.8804945054945055, 0.8539377289377289, 0.8818681318681318, 0.8713369963369964, 0.8717948717948718, 0.8713369963369964, 0.8571428571428571, 0.8713369963369964, 0.8704212454212454]
frequency_list = [0.5407509157509157, 0.6895604395604396, 0.8791208791208791, 0.9148351648351648, 0.9184981684981685, 0.9203296703296703, 0.9253663003663004, 0.8594322344322345, 0.9262820512820513, 0.9194139194139194, 0.9258241758241759, 0.924908424908425, 0.9097985347985348, 0.9203296703296703, 0.9207875457875457, 0.923992673992674, 0.9212454212454212, 0.9203296703296703, 0.9203296703296703, 0.9111721611721612]
cnn = [0.6968864, 0.82326007, 0.80448717, 0.8250916, 0.84386444, 0.8630952, 0.81959707, 0.86217946, 0.83836997, 0.86172163, 0.87225276, 0.85027474, 0.871337, 0.85347986, 0.86630034, 0.85805863, 0.86263734, 0.86172163, 0.8301282, 0.87362635]
print(np.max(cnn))
print(np.max(adjective_list))
list = [i for i in range(1,21)]
plt.plot(list,adjective_list ,color = "green",label = "Adjective words")
plt.plot(list,frequency_list,color = "red",label = "Word Frequency")
plt.plot(list,cnn,color = "blue",label = "GloVe word embedding")
plt.legend()
plt.title("COMPARING BETWEEN THREE STRATEGIES")
plt.xlabel('epoch', fontsize=10)
plt.ylabel('accuracy', fontsize=10)
plt.show()
