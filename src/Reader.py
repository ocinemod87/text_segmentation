from Fast_Text import Language_Model
from Model_Bidirectional import Models
import numpy as np
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import model_from_json


np.set_printoptions(suppress=True)

class Reader:

    def __init__(self):
        self.language_model = Language_Model()
        wv_matrix = np.load('/usr/src/app/wv_matrix_classifier_custom_big.npy')
        self.model_classifier = Models.Model_Bidirectional(197156, 100, wv_matrix, 100)
        self.model_classifier.load_weights('/usr/src/app/classlatest5.h5')


# # Open a file: file
# file = open('temp.md',mode='r')
#
# text_list = []
    def predict(self, file_name):
        try:
            file = open(file_name,mode='r')
            # read all lines at once
            lines = file.readlines()
            text_list = []

            for line in lines:
                #line = line.lower()
                text_list.append(line)
            # print(text_list)

            data = self.language_model.encode_text(text_list)


            print('LOADED')


            prediction = self.model_classifier.predict(data)

            print('Prediction for '+file_name[10:])
            result = (prediction > 0.1).sum()
            print('result '+str(result))

            if result > 0:
                x=0
                for line in prediction:
                    x+=1
                    if line > 0.1:
                        print('Line '+str(x))
                        print(text_list[(x-1)])
            else:
                print('Nothing found....')
        except:
            print("File "+file_name[10:]+" cannot be read.")
            pass

# selected = []
# lines_num = []
#
# for x, line in enumerate(prediction):
#     if line > 0.5:
#         selected.append(lines[x])
#         lines_num.append(x)

# # print(selected)
# new_data = Language_Model.encode_text(selected)
#
# prediction = model_multi.predict(new_data)
#
# print('------------------------------------------------------Splitting -------------------------------------------------------------')
# x = 0
# for line in prediction:
#     print(lines[lines_num[x]])
#     print(line)
#     x+=1
########################################################################################################################################################################




# y = 0
# x=0
#
# blocks = []
# block = []
# #blocks.append(block)
#
# for i in range(len(prediction)):
#     x+=1
#     if np.argmax(prediction[i], axis=0) == 1:
#         block = []
#         block.append(lines[i])
#         blocks.append(block)
#         print(x)
#         print(prediction[i])
#         #print(lines[i])
#     elif np.argmax(prediction[i], axis=0) == 2:
#         block.append(lines[i])
#         print(x)
#         print(prediction[i])
#     elif (np.argmax(prediction[i], axis=0) == 0) and (lines[i].strip == False):
#         block.append(lines[i])
#
# #
# # blocks = []
# # block = []
# # blocks.append(block)
# #
# # for i in range(len(prediction)):
# #     if np.argmax(prediction[i], axis=0) == 1:
# #         block = []
# #         block.append(lines[i])
# #         blocks.append(block)
# #         #print(lines[i])
# #     else:
# #         block.append(lines[i])
#
#
# print('BLOCKS')
# print(len(blocks))
#
# if not blocks[0]:
#     blocks.pop(0)
#
# y = 1
#
# # for block in blocks:
# #     print('-----------------BLOCK '+str(y)+'-----------------')
# #     y+=1
# #     for line in block:
# #         print(line)
#
# # x = 0
# # y = 1
# #
# # for block in blocks:
# #     print('-----------------BLOCK '+str(y)+'-----------------')
# #     data = now(block)
# #     prediction = model_classifier.predict(data)
# #     for line in prediction:
# #         x+=1
# #         print(x)
# #         print(line)
# #     y+=1
#
# #
# # if len(blocks) > 0:
# #     blocks.pop(0)
# #     for line in blocks[0]:
# #         print(line)
# # else:
# #     for line in block:
# #         print(line)
